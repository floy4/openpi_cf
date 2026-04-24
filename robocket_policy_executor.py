from collections import deque
from typing import List
from functools import partial
import numpy as np
import utils.logutil as logging
import random
import time
import cv2
import torch

from data.schema.episode_dataclass import Action, CameraFrame, Observation
from policy.Robocket.model.robocket import RobocketModel
from policy.Robocket.model.collator import RobocketCollator

from .base_policy_executor import BasePolicyExecutor

from .utils import SingleWorkerExecutor
from data.schema.dataloader3 import sample_history

class RobocketPolicyExecutor(BasePolicyExecutor):
    def __init__(
        self,
        model_path,
        pixel_keys: list = None,
        proprio_key: str = None,
        pixel_key_map: dict = None,
        proprio_key_map: dict = None,
        image_size=[256, 256],
        image_resize_size=None,
        action_decode_map: dict = None,
        detect_busy: bool = False,  # 是否检测任务繁忙（默认不检测，允许排队）
        enable_cf_eval: bool = False,  # 是否启用反事实评估（默认不启用）
        action_offset:int = 0,
        action_stride:int = 1,
        sucker_action_offset:int = 0,
        delay: dict = None,
        **kwargs,
    ) -> None:

        if pixel_key_map is None:
            pixel_key_map = {k: k for k in pixel_keys}
        if proprio_key_map is None:
            proprio_key_map = {proprio_key: proprio_key}
        self._pixel_key_map = pixel_key_map
        self._proprio_key_map = proprio_key_map
        self._action_decode_map = action_decode_map
        self.need_normalization_dict = {
            x["name"]: x.get("normalization", True)
            for proprio_list in self._proprio_key_map.values()
            for x in proprio_list
        }
        self.need_unnormalization_dict = {
            x["name"]: x["normalization"]
            for action_group_value in self._action_decode_map.values()
            for x in action_group_value
        }
        self._model_path = model_path
        if isinstance(image_size, (float, int)):
            image_size = (image_size, image_size)
        if image_resize_size is None:
            image_resize_size = {k: image_size for k in pixel_key_map}

        self._image_resize_size = image_resize_size
        self._action_offset = action_offset
        self._action_stride = action_stride
        self._sucker_action_offset = sucker_action_offset
        self._delay = delay
        self._detect_busy = detect_busy
        # 可选：启用反事实评估，仅在推理阶段用于分析 VLM vs proprio 的影响
        self._enable_cf_eval = enable_cf_eval
        # load finetuned model
        logging.info(f"Loading model {model_path}")
        self._model = RobocketModel.from_pretrained(model_path)
        self._enable_history_sampling = self._model.config.get("use_history",False)
        #获取history相关配置，仅在enable_history_sampling=True启用时使用,没有的话取默认值
        if self._enable_history_sampling:
            self._proprio_state_obs_window_size = self._model.config.get("proprio_state_obs_window_size",1)
            self._use_concise_history_prompt = self._model.config.get("use_concise_history_prompt",False)
            self._use_detailed_history_prompt = self._model.config.get("use_detailed_history_prompt",False)
            self._use_soft_history_prompt = self._model.config.get("use_soft_history_prompt",False)
            self._collate_fn = RobocketCollator(
                self._model.vlm_processor, image_order=pixel_keys, use_history=self._enable_history_sampling, use_concise_history_prompt=self._use_concise_history_prompt, use_detailed_history_prompt=self._use_detailed_history_prompt, use_soft_history_prompt=self._use_soft_history_prompt
            )
            logging.info(f"RobocketPolicyExecutor: enable history sampling with proprio_state_obs_window_size={self._proprio_state_obs_window_size}, use_history={self._enable_history_sampling}, use_concise_history_prompt={self._use_concise_history_prompt}, use_detailed_history_prompt={self._use_detailed_history_prompt}, use_soft_history_prompt={self._use_soft_history_prompt}")
        else:
            self._collate_fn = RobocketCollator(
                self._model.vlm_processor, image_order=pixel_keys
            )
        self._model = self._model.to(torch.bfloat16).to("cuda")
        self._model.eval()
        # 兼容前一版的历史窗口，仅在enable_history_sampling=False使用）
        self._history_horizon = self._model.config.get("obs_window_size",1)
        self._warm_up()

        logging.info("RobocketPolicy initialized!")

    def _warm_up(self):
        logging.info("RobocketPolicy Warming Up...")

        def random_image():
            return np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)

        obs = Observation(
            # self._pixel_key_map.values() looks like ["camera1_rgb", "camera12_rgb", ...]
            camera_frame=CameraFrame(
                **{k: random_image() for k in self._pixel_key_map.values()}
            ),
            arm1_joints_state=np.random.rand(6).tolist(),
            arm2_joints_state=np.random.rand(6).tolist(),
            arm1_gripper_state=np.random.rand(1).tolist(),
            arm2_gripper_state=np.random.rand(1).tolist(),
            arm1_sucker_state=np.random.rand(1).tolist(),
            arm2_sucker_state=np.random.rand(1).tolist(),
        )

        session = self.create_session()
        from tqdm import trange

        time_step = 1
        self.__call__(
            session, observation=obs, language_instruction="test", time_step=time_step
        )

        for _ in trange(10, desc="测速"):
            time_step += random.randint(1, 5)
            self.__call__(
                session,
                observation=obs,
                language_instruction="test",
                time_step=time_step,
            )

        logging.info("RobocketPolicy Warming Up Done!")

    def create_session(self):
        if self._enable_history_sampling:
            #NOTE: 根据sample_history所需的obs_num和stride均固定为4，表示3帧历史+1帧当前观测，如需修改，可在函数内调整，history buffer长度设置为obs_num×stride=16
            maxlen =  16

        else:
            maxlen = self._history_horizon
        return dict(
            historys=deque(maxlen=maxlen),
        )

    def update_session(
        self,
        session: dict,
        observation: Observation,
        language_instruction: str = None,
        time_step: int = 0, #新增
    ):

        observation = self.state_normalization(observation)
        inputs = {
            "images": {
                new_key: observation.camera_frame.__dict__[old_key]
                for new_key, old_key in self._pixel_key_map.items()
            },
            "states": {
                key: np.concatenate(
                    [
                        observation.__dict__[state_key["name"]]
                        for state_key in state_list
                    ]
                )
                for key, state_list in self._proprio_key_map.items()
            },
            "language_instruction": language_instruction,
            "time_step": time_step, #新增 用于时间步采样
        }

        inputs = self._observation_preprocess(inputs)

        session["historys"].append(inputs)

    def __call__(
        self,
        session: dict,
        observation: Observation,
        language_instruction: str = None,
        time_step: int = None,
        **kwargs,
    ) -> List[Action]:
        if self._delay is not None:
            start_time = time.time()
        assert isinstance(session, dict)

        # 当前观测存入历史队列，并更新当前观测
        self.update_session(session, observation, language_instruction, time_step)

        model_inputs = self.prepare_model_inputs(session)

        if self._enable_cf_eval:
            predict_actions = partial(
                self._predict_actions_with_cf,
                session=session,
                model_inputs=model_inputs,
                time_step=time_step,
            )
        else:
            predict_actions = partial(
                self._predict_actions,
                session=session,
                model_inputs=model_inputs,
                time_step=time_step,
            )

        # ===== 新增：detect_busy 模式（非阻塞轮询，不排队）=====
        if self._detect_busy:
            # 初始化excutor
            if "job_executor" not in session:
                worker = SingleWorkerExecutor()
                session["job_executor"] = worker
                # 首次：只提交，不取结果，返回空动作
                worker.submit(predict_actions)
                actions, result_time_step = [], time_step
            else:
                worker: SingleWorkerExecutor = session["job_executor"]
                results = worker.result()
                if results is None:
                    actions, result_time_step = [], time_step
                else:
                    actions, result_time_step = results[0],results[1]
                    worker.submit(predict_actions)
        else:
            actions, result_time_step = predict_actions()
        # 4) 延迟模拟,用于debug，将推理计算快的模型调慢，来验证慢的模型是因为推理计算慢导致的效果差，还是预测精度不够
        if self._delay is not None:
            cost  = time.time()-start_time
            delay = 0.001 * random.gauss(self._delay.mean,self._delay.std)
            time.sleep(max(0, delay - cost ))

        return actions,result_time_step

    def _clone_model_inputs(self, model_inputs):
        """浅拷贝一份 model_inputs，用于构造反事实输入，避免修改原始 batch。

        model_inputs 预期是 dict，其中包含 tensor/ndarray 等，可安全复用底层存储，
        这里只对顶层和嵌套 dict 做浅拷贝。
        """
        if model_inputs is None:
            return None
        cloned = {}
        for k, v in model_inputs.items():
            if isinstance(v, dict):
                cloned[k] = {kk: vv for kk, vv in v.items()}
            else:
                cloned[k] = v
        return cloned

    def _make_cf_inputs_proprio_zero(self, model_inputs):
        """构造反事实输入：将 proprio（states）置零，保留 VLM 相关输入不变。

        要求 collator 输出的 `model_inputs` 中必须包含名为 "proprioception" 的张量字段；
        若不存在或类型不符合预期，将直接抛出异常，避免静默失败。
        """
        cf_inputs = self._clone_model_inputs(model_inputs)
        if cf_inputs is None:
            logging.warning("_make_cf_inputs_proprio_zero: model_inputs is None, skip cf proprio zero.")
            return None
        if "proprioception" not in cf_inputs:
            logging.warning(
                "_make_cf_inputs_proprio_zero: 'proprioception' key not found in model_inputs, skip cf proprio zero."
            )
            return None

        v = cf_inputs["proprioception"]
        if not isinstance(v, torch.Tensor):
            logging.warning(
                f"_make_cf_inputs_proprio_zero: expected 'proprioception' to be torch.Tensor, got {type(v)}; skip cf proprio zero."
            )
            return None
        cf_inputs["proprioception"] = torch.zeros_like(v)
        return cf_inputs

    def _make_cf_inputs_image_zero(self, model_inputs):
        """构造反事实输入：将图像输入置为零或均值，保留 proprio 不变。

        要求 collator 输出的 `model_inputs` 中必须包含图像字段：
        - 优先使用 "pixel_values"；若不存在则使用 "images"；
        若两者都不存在或类型不符合预期，将直接抛出异常，避免静默失败。
        """
        cf_inputs = self._clone_model_inputs(model_inputs)
        if cf_inputs is None:
            logging.warning("_make_cf_inputs_image_zero: model_inputs is None, skip cf image zero.")
            return None
        # 优先 pixel_values，其次 images
        if "pixel_values" in cf_inputs:
            key = "pixel_values"
        elif "images" in cf_inputs:
            key = "images"
        else:
            logging.warning(
                "_make_cf_inputs_image_zero: neither 'pixel_values' nor 'images' found in model_inputs, skip cf image zero."
            )
            return None

        v = cf_inputs[key]
        if isinstance(v, torch.Tensor):
            cf_inputs[key] = torch.zeros_like(v)
        elif isinstance(v, dict):
            # 针对多相机 / 多通道字典结构：逐个张量置零
            new_dict = {}
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    new_dict[kk] = torch.zeros_like(vv)
                else:
                    logging.warning(
                        f"_make_cf_inputs_image_zero: value for key '{kk}' under '{key}' must be torch.Tensor, got {type(vv)}; skip cf image zero."
                    )
                    return None
            cf_inputs[key] = new_dict
        else:
            logging.warning(
                f"_make_cf_inputs_image_zero: expected '{key}' to be torch.Tensor or dict of tensors, got {type(v)}; skip cf image zero."
            )
            return None
        # 为了严格评估 image-only effect，清空 language_instruction（如果存在），
        # 避免 language prompt 将信息泄露到 image-zero 的反事实评估中。
        if "language_instruction" in cf_inputs:
            try:
                cf_inputs["language_instruction"] = ""
            except Exception:
                cf_inputs["language_instruction"] = None

        return cf_inputs

    def _compute_action_diff(self, actions_base, actions_cf):
        """计算两组动作之间的差异标量，用于评估模态 effect。

        简单做法：将各个关节/夹爪动作拼接后，计算 L2 距离的均值。
        """
        if not actions_base or not actions_cf:
            return 0.0
        n = min(len(actions_base), len(actions_cf))
        diffs = []
        for i in range(n):
            a = actions_base[i].__dict__
            b = actions_cf[i].__dict__
            vec_a = []
            vec_b = []
            for k in a.keys():
                va = a[k]
                vb = b.get(k, None)
                if va is None or vb is None:
                    continue
                try:
                    va_arr = np.array(va, dtype=np.float32).reshape(-1)
                    vb_arr = np.array(vb, dtype=np.float32).reshape(-1)
                    m = min(len(va_arr), len(vb_arr))
                    vec_a.append(va_arr[:m])
                    vec_b.append(vb_arr[:m])
                except Exception:
                    continue
            if not vec_a:
                continue
            va_cat = np.concatenate(vec_a)
            vb_cat = np.concatenate(vec_b)
            diff = np.linalg.norm(va_cat - vb_cat)
            diffs.append(diff)
        if not diffs:
            return 0.0
        return float(np.mean(diffs))

    def _predict_actions_with_cf(
        self,
        session,
        model_inputs,
        time_step: int = None,
        reweight_action_with_cf: bool = True,
        cf_guidance_scale: float = 0.1,
        vlm_effect_upper_threshold: float = 0.5,
    ):
        """带反事实评估的预测：

        - baseline: 正常输入 → a_base
        - CF_proprio_zero: 将 proprio 置零 → a_cf_prop0
        - CF_image_zero: 将图像置零 → a_cf_img0

        如果 reweight_action_with_cf=True，则利用 CF 计算出的 effect 对最终动作进行加权修正 (Classifier-Free Guidance 思想)。
        公式: Action_Final = Action_Base + scale * (w_vlm * (Action_Base - Action_ImgZero) + w_prop * (Action_Base - Action_PropZero))
        """
        logging.debug(f"-----------------------------[bg] start predict_with_cf t={time_step}")
        start = time.time()
        original_model_inputs = model_inputs.copy()
        with torch.inference_mode():
            # baseline
            actions_base_raw = self._model.predict_actions(model_inputs)
            actions_base_tensor = actions_base_raw[0]
            actions_base = self.convert_action(actions_base_tensor)
            actions_base = self._action_post_process(actions_base)

            # CF: zero proprio
            cf_inputs_prop0 = self._make_cf_inputs_proprio_zero(original_model_inputs)
            raw_prop0 = None
            if cf_inputs_prop0 is not None:
                try:
                    raw_prop0 = self._model.predict_actions(cf_inputs_prop0)[0]
                    actions_prop0 = self.convert_action(raw_prop0)
                    actions_prop0 = self._action_post_process(actions_prop0)
                    effect_prop = self._compute_action_diff(actions_base, actions_prop0)
                except Exception as e:
                    logging.warning(f"CF eval (proprio_zero) failed at t={time_step}: {e}")
                    effect_prop = 0.0
            else:
                effect_prop = 0.0

            # CF: zero images
            cf_inputs_img0 = self._make_cf_inputs_image_zero(original_model_inputs)
            raw_img0 = None
            if cf_inputs_img0 is not None:
                try:
                    raw_img0 = self._model.predict_actions(cf_inputs_img0)[0]
                    actions_img0 = self.convert_action(raw_img0)
                    actions_img0 = self._action_post_process(actions_img0)
                    effect_vlm = self._compute_action_diff(actions_base, actions_img0)
                except Exception as e:
                    logging.warning(f"CF eval (image_zero) failed at t={time_step}: {e}")
                    effect_vlm = 0.0
            else:
                effect_vlm = 0.0
            
            # Reweight Logic: fuse modality deltas proportionally to their CF effects
            if reweight_action_with_cf:
                raw_base = actions_base_raw[0]
                raw_final = {}
                # def to_np(x):
                #     if isinstance(x, torch.Tensor):
                #         return x.detach().cpu().numpy()
                #     return np.array(x)

                # compute modality importance weights from effects or respect explicit cf_mode
                # eps = 1e-8
                # mode = getattr(self, "_cf_mode", "adaptive")

                # base: no CF, return baseline early
                # if mode == "base":
                #     return actions_base, time_step

                # if mode == "vlm-only":
                #     w_vlm, w_prop = 1.0, 0.0
                # else:
                #     total_effect = float(2*effect_vlm + 2*effect_prop) + eps
                #     w_vlm = float(effect_vlm) / total_effect
                #     w_prop = float(effect_prop) / total_effect
                #     w_base = float(effect_vlm + effect_prop) / total_effect
                # if float(effect_vlm) > vlm_effect_upper_threshold:
                #     logging.info(
                #             f"base t={time_step} effect_vlm={effect_vlm:.3f}")
                #     return actions_base, time_step

                # If any CF output is missing, skip reweighting to avoid unsafe indexing
                for k, v_base in raw_base.items():
                    # compute deltas (base - ablated)
                    delta_img = v_base - raw_img0[k]
                    delta_prop = v_base - raw_prop0[k]
                    # No clipping: apply combined delta scaled by guidance scale
                    #v_final = v_base + scale * combined_delta 
                    #v_final = (1-scale)*v_base + scale * delta_img 不如直接v_base为1的
                    #v_final = v_base + scale*delta_img 效果还行
                    #v_final= v_base + scale*delta_prop 极度扭曲,无法正常完成任务
                    #v_final = w_base * v_base + w_vlm * delta_img + w_prop * delta_prop 不扭曲、悬空不会干活
                    #v_final = v_base + scale*(w_base * v_base + w_vlm * delta_img + w_prop * delta_prop) 扭曲严重,无法正常完成任务
                    
                    # 改进方案：对 delta_prop 进行归一化或使用小权重
                    # 方案A: 只用图像引导，忽略 proprio（最保守，基于方式2的良好效果）
                                        # if float(effect_vlm) > float(vlm_effect_upper_threshold):
                    #     raw_final[k] = v_base
                    #     mode = "base"
                    #raw_final[k] = v_base + cf_guidance_scale * delta_img #平铺下可以正常完成叠衣,但是看不太到相比于直接使用action base的提升
                    #raw_final[k] = (1-cf_guidance_scale)*v_base + cf_guidance_scale * raw_img0[k]
                    # 方案B: 对 delta_prop 使用很小的权重（0.05-0.1），避免扭曲
                    # prop_scale = 0.05  # 可调节，建议0.01-0.1之间
                    # v_final = v_base + scale * delta_img + prop_scale * delta_prop 不能正常完成任务
                    
                    # 方案C: 归一化 delta_prop 后再使用（限制其影响范围）
                    # 兼容 torch.Tensor 和 numpy.ndarray
                    # 动作不扭曲,但是精度没有提升,结束时候还会引入机械臂的摇摆,不能很快的恢复到初始状态
                    # if isinstance(delta_prop, torch.Tensor):
                    #     abs_max = torch.abs(delta_prop).max()
                    #     delta_prop_norm = delta_prop / (abs_max + 1e-6) if abs_max > 0 else delta_prop
                    # else:
                    #     abs_max = np.abs(delta_prop).max()
                    #     delta_prop_norm = delta_prop / (abs_max + 1e-6) if abs_max > 0 else delta_prop
                    # v_final = v_base + scale * (w_vlm * delta_img + w_prop * 0.1 * delta_prop_norm)
                    
                    # 方案D: clip delta_prop 避免极端值（当前最优）
                    # 平铺叠衣服下可以正常完成任务,并避免掉落
                    # 混乱下bash0, 48
                    # if isinstance(delta_prop, torch.Tensor):
                    #     delta_prop_clipped = torch.clamp(delta_prop, -0.1, 0.1)
                    # else:
                    #     delta_prop_clipped = np.clip(delta_prop, -0.1, 0.1)
                    # v_final = v_base + scale * delta_img + 0.05 * delta_prop_clipped
                    
                    # 方案E: 自适应权重 - 根据effect比例动态调整prop权重（建议0.02-0.1之间）
                    #effect_prop越大，权重越高, 效果上与方案D相似,平铺下没有问题,混乱叠衣服会出现左右摇摆的死循环态,68
                        # Decide whether to apply CF reweighting based on effect magnitudes
                    #total_effect = float(effect_vlm + effect_prop)
                    # If VLM effect is unexpectedly large, prefer baseline (avoid overconfident VLM-driven changes)
                    if float(effect_vlm) > float(vlm_effect_upper_threshold):
                        raw_final[k] = v_base
                        mode = "base"
                    # elif total_effect < float(min_total_effect) or float(effect_vlm) < float(min_vlm_effect):
                    #     # Effects too small — keep baseline for stability
                    #     raw_final[k] = v_base
                    #     mode = "base1"
                    else:
                        prop_weight = 0.05 * min(1.0, effect_prop / (effect_vlm + 1e-6))  # effect_prop越大，权重越高
                        if isinstance(delta_prop, torch.Tensor):
                            delta_prop_clipped = torch.clamp(delta_prop, -0.1, 0.1)
                        else:
                            delta_prop_clipped = np.clip(delta_prop, -0.1, 0.1)
                        v_final = v_base + cf_guidance_scale * delta_img + prop_weight * delta_prop_clipped
                        raw_final[k] = v_final
                        mode = "CF"
                    
                    # 方案F: 温和限制 - 使用tanh平滑限制而非硬clip，保留更多信息
                    # 平铺没有方案E好,掉落情况仍然多,pass
                    # if isinstance(delta_prop, torch.Tensor):
                    #     delta_prop_soft = torch.tanh(delta_prop / 0.1) * 0.1  # tanh将输入映射到[-0.1, 0.1]
                    # else:
                    #     delta_prop_soft = np.tanh(delta_prop / 0.1) * 0.1
                    # v_final = v_base + scale * delta_img + 0.05 * delta_prop_soft
                    
                    # 方案G: 自适应clip范围 - 根据delta_img幅度调整delta_prop范围，保持两者协调
                    # 平铺情况下夹的深的优势没有了,pass
                    # if isinstance(delta_img, torch.Tensor):
                    #     img_magnitude = torch.abs(delta_img).mean()
                    #     clip_range = float(max(0.05, min(0.2, img_magnitude * 0.5)))  # 范围在[0.05, 0.2]之间
                    # else:
                    #     img_magnitude = np.abs(delta_img).mean()
                    #     clip_range = float(max(0.05, min(0.2, img_magnitude * 0.5)))
                    # if isinstance(delta_prop, torch.Tensor):
                    #     delta_prop_clipped = torch.clamp(delta_prop, -clip_range, clip_range)
                    # else:
                    #     delta_prop_clipped = np.clip(delta_prop, -clip_range, clip_range)
                    # v_final = v_base + scale * delta_img + 0.05 * delta_prop_clipped
                #
                actions_final = self.convert_action(raw_final)
                actions_final = self._action_post_process(actions_final)
                actions_base = actions_final

        cost = time.time() - start
        logging.info(
            f"scale={cf_guidance_scale} t={time_step} cost={cost:.3f}s, effect_vlm={effect_vlm:.3f}"
        )
        logging.debug(f"----------------------------[bg] end predict_with_cf t={time_step}, cost={cost:.3f}s")
        
        return actions_base, time_step

    def prepare_model_inputs(self, session: dict) -> dict:
        """
        从 session['historys'] 采样构建模型输入：
        - sample_history 已内置“历史不足时用当前步 padding”,固定3帧历史+1帧当前，详见 dataloader3.sample_history
        - 仅对 image 进行采样，state 直接取当前步
        """
        historys = session["historys"]
        # === 新版：按采样窗口/步长（sample_history 已经在顶部通过dataloader3导入） ===
        if self._enable_history_sampling:
            buf = list(historys)
            if not buf:
                raise ValueError("No observations available to build model inputs.")
            n = len(buf)
            i = n - 1  # 当前步索引
            # NOTE: 准备image历史观测，sample_history所需的obs_num和stride均固定为4，表示3帧历史+1帧当前观测，如需修改，可在函数内调整
            # 根据当前实验效果，仅采用image history效果最好，因此目前只对image进行采样，后续如需增加对robot state的采样，可与image_indices采用相同的方式生成frame的index。
            # 如增加robot state采样，优先将image的采样index赋值给robot state，保持两者对齐。但两者分别送入不同的encoder网络，也可以各自独立采样，建议robot state的采样频率大于等于image 的采样频率。
            image_indices = sample_history(time_step=i)
            image_keys = list(buf[-1]["images"].keys())
            state_keys = list(buf[-1]["states"].keys())

            image_frames = [buf[j] for j in image_indices]
            state_frames = [buf[i]]

            images_stacked = {
                k: np.stack([f["images"][k] for f in image_frames]) for k in image_keys
            }
            states_stacked = {
                k: np.stack([f["states"][k] for f in state_frames]) for k in state_keys
            }
            if self._use_detailed_history_prompt or self._use_soft_history_prompt:
                images_idx_stacked = [buf[j]["time_step"] for j in image_indices]
                inputs = {
                    "images": images_stacked,
                    "states": states_stacked,
                    "language_instruction": buf[-1]["language_instruction"],
                    "images_idx": images_idx_stacked,
                }
            else:
                inputs = {
                    "images": images_stacked,
                    "states": states_stacked,
                    "language_instruction": buf[-1]["language_instruction"],
                }
        else:
            #todo 整理history代码，视history效果看是否保留
            if len(historys) < self._history_horizon:
                historys = [historys[0]] * (self._history_horizon - len(historys)) + [obs for obs in historys]
            inputs = {
                "images": {k: np.stack([obs["images"][k] for obs in historys]) for k in historys[0]["images"]},
                "states": {k: np.stack([obs["states"][k] for obs in historys]) for k in historys[0]["states"]},
                "language_instruction": historys[-1]["language_instruction"],
            }
        return self._collate_fn([inputs])[0]

    def _predict_actions(
        self,
        session,
        model_inputs,
        time_step: int = None,
    ):
        logging.info(f"-----------------------------[bg] start predict t={time_step}")
        start = time.time()
        with torch.inference_mode():
            actions = self._model.predict_actions(model_inputs)
        actions = actions[0]  # batch 中只有一个
        actions = self.convert_action(actions)
        actions = self._action_post_process(actions)
        actions = actions[self._action_offset::self._action_stride]
        sucker_action_offset = max(0,self._sucker_action_offset-self._action_offset)//self._action_stride
        for i,a in enumerate(actions):
            a: Action
            b = actions[min(i+sucker_action_offset,len(actions)-1)]
            a.arm1_sucker_action_abs=b.arm1_sucker_action_abs
            a.arm2_sucker_action_abs=b.arm2_sucker_action_abs
        cost = time.time() - start
        logging.info(f"----------------------------[bg] end predict t={time_step}, cost={cost:.3f}s")
        logging.info(f"Robocket Prediction Cost={cost:.3f}s")
        return actions, time_step

    def state_normalization(self, observation: Observation):
        """
        The dataset statistics infomation is generated from the dataset, assigned to the model during training, and saved as a JSON file alongside the model checkpoint.
        For detailed information, please refer to `policy.Robocket.scripts.finetune.py` and `policy.Robocket.model.robocket.py`.
        The content of the dataset statistics information is a `dict` object that appears as follows:
        ```
        {
            "action": {
                "mean": {
                    "arm1_joints_action_abs": [
                        -0.31523263454437256,
                        ...
                    ],
                    "arm1_gripper_action_abs": [
                        0.20922365188598632
                    ],
                    "arm2_joints_action_abs": [
                        0.25583967566490173,
                        ...
                    ],
                    "arm2_gripper_action_abs": [
                        0.16857749223709106
                    ]
                },
                "std": {
                    "arm1_joints_action_abs": [
                        0.24571500718593597,
                        ...
                    ],
                    "arm1_gripper_action_abs": [
                        0.3329158306121826
                    ],
                    "arm2_joints_action_abs": [
                        0.25338831543922424,
                        ...
                    ],
                    "arm2_gripper_action_abs": [
                        0.29949469566345216
                    ]
                }
            },
            "state": {
                "mean": {
                    "arm1_joints_state": [
                        -0.31483012437820435,
                        ...
                    ],
                    "arm1_gripper_state": [
                        0.29569182395935056
                    ],
                    "arm2_joints_state": [
                        0.254569411277771,
                        ...
                    ],
                    "arm2_gripper_state": [
                        0.24195194244384766
                    ]
                },
                "std": {
                    "arm1_joints_state": [
                        0.24536313116550446,
                        ...
                    ],
                    "arm1_gripper_state": [
                        0.34794049263000487
                    ],
                    "arm2_joints_state": [
                        0.25301501154899597,
                        ...
                    ],
                    "arm2_gripper_state": [
                        0.32497851848602294
                    ]
                }
            }
        }
        ```
        """
        eps = 1e-6
        metadata = self._model.dataset_statistics["state"]
        # 从 proprio_key_map 配置中提取所有状态属性
        # proprio_key_map 结构示例：{"proprio": [{"name": "arm1_joints_state", ...}, ...]}
        # 使用集合存储，避免重复的属性名
        state_att = {t for _, v in self._proprio_key_map.items() for t in v}
        # 遍历 observation 数据的所有属性: 机械臂关节状态 夹爪状态 机械臂姿态 ...
        for k, v in observation.__dict__.items():
            if v is None:
                continue
            if k in metadata["std"]:
                std = metadata["std"][k]
                mean = metadata["mean"][k]
                # 检查该属性在配置中是否指定了 selected_indices（需提取的特定维度）
                for att in state_att:
                    if att["name"] == k and att.get("selected_indices") is not None:
                         v, std, mean = (
                            [arr[dim] for dim in att["selected_indices"]]
                            for arr in (v, std, mean)
                        )
                # 根据配置决定是否进行归一化（默认启用）
                normalization = self.need_normalization_dict.get(k, True)
                observation.__dict__[k] = np.where(
                    normalization, (np.array(v) - np.array(mean)) / (np.array(std) + eps), np.array(v)
                )
        return observation

    def action_unnormalization(self, action: Action):
        metadata = self._model.dataset_statistics["action"]
        for k, v in action.__dict__.items():
            if v is None:
                continue
            if k in metadata["std"]:
                std = metadata["std"][k]
                mean = metadata["mean"][k]
                normalization = self.need_unnormalization_dict.get(k, True)
                action.__dict__[k] = np.where(
                    normalization, (np.array(v) * std + mean), np.array(v)
                ).tolist()
        return action

    def _observation_preprocess(self, obs):

        def resize_image(rgb_array: np.ndarray, size: tuple) -> np.ndarray:
            original_size, expected_size = tuple(rgb_array.shape[:2]), tuple(size[:2])
            if original_size != expected_size:
                rgb_array = cv2.resize(
                    rgb_array, expected_size[::-1], interpolation=cv2.INTER_LINEAR
                )
                assert tuple(rgb_array.shape[:2]) == expected_size
            return rgb_array

        curr_obs = obs
        resized = {
            k: resize_image(curr_obs["images"][k], size)
            for k, size in self._image_resize_size.items()
        }
        curr_obs["images"].update(resized)
        for k in self._pixel_key_map:
            curr_obs["images"][k] = np.array(curr_obs["images"][k])

        return curr_obs

    def _get_dataset_statistics(self):
        return self._model.dataset_statistics

    def convert_action(self, actions: dict[str, np.ndarray]) -> Action:
        groups = {}
        for group_name, group_value in self._action_decode_map.items():
            start_idx = 0
            for x in group_value:
                dim = x["dim"]
                end_idx = start_idx + dim
                groups[x["name"]] = np.array(actions[group_name])[:, start_idx:end_idx]
                start_idx = end_idx
        chunk_size = list(groups.values())[0].shape[0]
        return [
            Action(**{k: v[i, :].tolist() for k, v in groups.items()})
            for i in range(chunk_size)
        ]

    def action_clip(self, actions):
        # TODO: 实现clip
        return actions

    def _action_post_process(self, actions):
        actions = [self.action_unnormalization(action) for action in actions]
        actions = [self.action_clip(action) for action in actions]
        return actions
