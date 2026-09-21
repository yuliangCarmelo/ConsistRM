# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import copy
import os
import socket
import re
import hydra
import ray
import torch
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import logging
import math
import numpy as np
import random
import requests
import json
import time

from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.sampler import AbstractSampler
from verl.utils.import_utils import load_extern_type
from verl import DataProto
# from sentence_transformers import SentenceTransformer

class RewardManager():
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine, format_weight=1, answer_weight=1, alpha=1) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.format_weight = format_weight
        self.answer_weight = answer_weight
        self.alpha = alpha
        # self.embedding_model = SentenceTransformer(
        #     "Qwen/Qwen3-Embedding-4B",
        #     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        #     tokenizer_kwargs={"padding_side": "left"},
        # )

    def extract_score(self, resp):
        result_match = re.search(r'<Result>(.*?)</Result>', resp, re.DOTALL | re.IGNORECASE)
        if not result_match:
            return None

        result_content = result_match.group(1).strip()
        
        if "Response 1 is better than Response 2" in result_content:
            return -1
        elif "Response 2 is better than Response 1" in result_content:
            return 1
        else: 
            return None

    def check_format(self, resp):
        has_criterion = re.search(r'<Criterion>.*?</Criterion>', resp, re.DOTALL) is not None
        has_analysis = re.search(r'<Analysis>.*?</Analysis>', resp, re.DOTALL) is not None
        has_result = re.search(r'<Result>.*?</Result>', resp, re.DOTALL) is not None

        if not (has_criterion and has_analysis and has_result):
            print("[DEBUG] 缺少必要的标签：",
                f"Criterion={has_criterion}, Analysis={has_analysis}, Result={has_result}")
            return False, None, None

        criterion_count = len(re.findall(r'<Criterion>.*?</Criterion>', resp, re.DOTALL))
        analysis_count = len(re.findall(r'<Analysis>.*?</Analysis>', resp, re.DOTALL))
        result_count = len(re.findall(r'<Result>.*?</Result>', resp, re.DOTALL))

        # 必须成对出现且仅有一对
        if criterion_count != 1 or analysis_count != 1 or result_count != 1:
            print(f"[DEBUG] 标签数量错误：Criterion={criterion_count}, "
                f"Analysis={analysis_count}, Result={result_count}")
            return False, None, None

        # 保证 </Result> 是最后的输出（后面只能跟 <|im_end|>）
        if '</Result>' not in resp:
            print("[DEBUG] 未找到 </Result> 标签")
            return False, None, None
        after_result = resp.split('</Result>')[-1].strip()
        if not after_result.startswith('<|im_end|>'):
            print(f"[DEBUG] </Result> 后面不合法，剩余内容开头是: {after_result[:50]!r}")
            return False, None, None

        # 检查 Result 内容是否合法
        result_match = re.search(r'<Result>(.*?)</Result>', resp, re.DOTALL)
        result_content = result_match.group(1).strip() if result_match else ""

        valid_result = (
            "Response 1 is better than Response 2" in result_content or
            "Response 2 is better than Response 1" in result_content
        )
        if not valid_result:
            print(f"[DEBUG] <Result> 内容不合法: {result_content!r}")

            return False, None, None
        
        # 提取<Analysis>和<Criterion>标签内的内容
        criterion_match = re.search(r'<Criterion>(.*?)</Criterion>', resp, re.DOTALL)
        analysis_match = re.search(r'<Analysis>(.*?)</Analysis>', resp, re.DOTALL)

        # 去除内容前后的空白字符
        criterion_content = criterion_match.group(1).strip() if criterion_match else ""
        analysis_content = analysis_match.group(1).strip() if analysis_match else ""
       

        return True, criterion_content, analysis_content

    def call_embed_service(self, texts):
        url = "http://127.0.0.1:8000/embed"
        headers = {"Content-Type": "application/json"}
        
        start_total = time.time()

        request_body = {
            "texts": texts,
            "max_length": 1024
        }
        print("texts", texts)
        print("request_body", request_body)
        try:
            start_request = time.time()
            prepare_cost = start_request - start_total
            print(f"<debug>embed_service: 请求准备耗时 {prepare_cost:.4f} 秒（文本数量：{len(texts)}）")
            
            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(request_body)
            )

            end_request = time.time()
            request_cost = end_request - start_request
            print(f"<debug>embed_service: 接口请求耗时 {request_cost:.4f} 秒（状态码：{response.status_code}）")
        
            
            if response.status_code == 200:

                end_total = time.time()
                total_cost = end_total - start_total
                print(f"<debug>embed_service: 总耗时 {total_cost:.4f} 秒（成功返回，文本数量：{len(texts)}）")
                print("<debug>response: ", response.json())
                return response.json()
            else:
                print(f"服务请求失败！状态码：{response.status_code}")
                print(f"错误详情：{response.text}")
                return None
                
        except Exception as e:
            end_total = time.time()
            total_cost = end_total - start_total
            print(f"<debug>embed_service: 调用异常，总耗时 {total_cost:.4f} 秒（错误：{str(e)}）")
            print(f"服务调用出错：{str(e)}")
            return None

    # def calculate_similarity(self, test_list):
    #     if not test_list:
    #         print("[WARNING] 输入文本为空，无法计算相似度")
    #         return None
        
    #     result = self.call_embed_service(test_list)
    #     print("<debug> sim result:", result)
    #     #这里返回的是一个字典的矩阵
    #     return result

    # #众数作为baseline
    # def get_group_baseline(self, scores):
    #     if not scores:
    #         return None
        
    #     # 统计每个分数的出现频率
    #     count = defaultdict(int)
    #     for s in scores:
    #         count[s] += 1
        
    #     frequencies = list(count.values())
    #     max_freq = max(frequencies)
    
    #     # 提取所有众数
    #     modes = [s for s, freq in count.items() if freq == max_freq]
        
    #     # 情况1：所有分数出现次数相同（所有分数都是众数）取中位数
    #     if len(modes) == len(count):
    #         sorted_scores = sorted(scores)
    #         n = len(sorted_scores)
    #         mid = n // 2
    #         if n % 2 == 1:
    #             return sorted_scores[mid]
    #         else:
    #             return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        
    #     # 情况2：只有一个众数
    #     if len(modes) == 1:
    #         return modes[0]

    #     # 情况3：多个众数，返回这些众数的中位数
    #     else:
    #         sorted_modes = sorted(modes)
    #         n_modes = len(sorted_modes)
    #         mid_mode = n_modes // 2
    #         if n_modes % 2 == 1:
    #             return sorted_modes[mid_mode]
    #         else:
    #             return (sorted_modes[mid_mode - 1] + sorted_modes[mid_mode]) / 2
    def get_current_baseline(self, processed_items):
        """
        遍历 processed_items，为每个元素计算并添加
        epoch_level_confidence, group_level_confidence, current_baseline
        """
        for item in processed_items:
            # 取出两个列表
            uid = item["uid"]
            print("<debug>uid", uid)
            uid_x = item["uid_x"]
            print("<debug>uid", uid_x)
            expriment_scores = item.get("expriment_score_list", [])
            print("<debug>expriment_scores", expriment_scores)
            rollout_score_list = item.get("rollout_score_list", {})
            print("<debug>rollout_score_list", rollout_score_list)
            if rollout_score_list:
                rollout_scores = rollout_score_list[uid]
                print("<debug>rollout_scores", rollout_scores)

                # nonzero_count = sum(1 for x in expriment_scores if x != 0)
                # print("<debug>nonzero_count", nonzero_count)
                
                # 计算 epoch_level_confidence
                # if expriment_scores:
                #     epoch_level_confidence = sum(int(x) for x in expriment_scores) / int(nonzero_count)
                if expriment_scores:
                    epoch_level_confidence = sum(int(x) for x in expriment_scores) / len(expriment_scores)
                else:
                    epoch_level_confidence = 0 
                # 计算 group_level_confidence
                if rollout_scores:
                    group_level_confidence = sum(int(x) for x in rollout_scores) / len(rollout_scores)
                else:
                    group_level_confidence = 0

                # 两者相加得到总值
                total_confidence = epoch_level_confidence + group_level_confidence
                print("<debug>epoch_level_confidence:", epoch_level_confidence)
                print("<debug>group_level_confidence:", group_level_confidence)
                print("<debug>total_confidence:", total_confidence)

                # 根据正负性确定 baseline
                if total_confidence > 0:
                    current_baseline = 1
                elif total_confidence < 0:
                    current_baseline = -1
                else:
                    current_baseline = 0

                print("<debug>current_baseline:", current_baseline)

                item["epoch_level_confidence"] = epoch_level_confidence
                item["group_level_confidence"] = group_level_confidence
                item["current_baseline"] = current_baseline
            
            else:
                item["epoch_level_confidence"] = None
                item["group_level_confidence"] = None
                item["current_baseline"] = None

        return processed_items

    def build_uid_baseline_map(self, processed_items):
        """
        从 processed_items 提取 {uid: current_baseline} 的映射
        """
        uid_baseline_map = {}
        for item in processed_items:
            uid = item.get("uid")
            baseline = item.get("current_baseline")
            if uid is not None:
                uid_baseline_map[uid] = baseline
        print("<debug>uid_baseline_map:", uid_baseline_map)
        
        return uid_baseline_map

    def get_group_baseline(self, scores):
        if not scores:
            return None 
        # 统计-1和1的出现次数
        count_1 = scores.count(1)
        count_m1 = len(scores) - count_1
        
        # 比较两个数字的出现次数
        if count_m1 > count_1:
            return -1
        elif count_1 > count_m1:
            return 1
        else:
            # 出现次数相同，存在多个众数，返回0
            return 0
    
    # 计算置信度
    def compute_confidence(self, logprobs):
        """
        从 logprobs 张量计算单个样本的置信度分数
        Args:
            logprobs: 形状为 [seq_len] 的张量，每个元素是 token 的对数概率（单个样本）
        Returns:
            该样本的置信度分数
        """
        epsilon = 1e-9
        
        token_logprobs = logprobs
        print("<debug>token_logprobs:", token_logprobs)
        print("<debug>len(token_logprobs):", len(token_logprobs))

        # # 过滤接近0的无效对数概率（避免数值问题）
        # valid_mask = torch.abs(token_logprobs) > epsilon
        # valid_logprobs = token_logprobs[valid_mask]
        # print("<debug>valid_logprobs:", valid_logprobs)
        # print("<debug>len(valid_logprobs):", len(valid_logprobs))

        # if valid_logprobs.numel() > 0: 
        #     # 计算有效token的平均对数概率（衡量整体置信度）
        #     mean_logprob = valid_logprobs.mean().item()
        #     print("<debug>mean_logprob:", mean_logprob)

        #     # 保留3位小数作为置信度（对数概率越接近0，置信度越高）
        #     seq_confs = round(mean_logprob, 5)
        # else:
        #     # 无有效token时默认置信度为0
        #     seq_confs = 0.0
        if token_logprobs.numel() > 0: 
            # 计算有效token的平均对数概率（衡量整体置信度）
            mean_logprob = token_logprobs.mean().item()
            print("<debug>mean_logprob:", mean_logprob)

            # 保留5位小数作为置信度（对数概率越接近0，置信度越高）
            # seq_confs = round(mean_logprob, 5)
            seq_confs = mean_logprob
        else:
            # 无有效token时默认置信度为0
            seq_confs = 0.0
    
        return seq_confs
    
    # 找到回复中对应位置的mask
    # def mask_after_response(self, response, tokenizer, start_tag="<Result>", end_tag="</Result>"):
    #     """
    #     在序列中找到 <Result> ... </Result> 的范围，并把 </Result> 之后的内容全部mask掉
    #     Args:
    #         response: torch.Tensor, shape (seq_len,)
    #         tokenizer: HuggingFace tokenizer
    #         start_tag: str, 默认 "<Result>"
    #         end_tag: str, 默认 "</Result>"
    #     Returns:
    #         masked_response: torch.Tensor, 已经mask掉</Result>之后的部分
    #         mask: torch.Tensor, 0/1 mask，和response等长
    #         positions: dict，包含 start_pos, end_pos
    #     """
    #     # 标签对应的token_id序列
    #     start_ids = self.tokenizer.encode(start_tag, add_special_tokens=False)
    #     end_ids   = self.tokenizer.encode(end_tag, add_special_tokens=False)
    #     print("<debug>start_ids:",start_ids)
    #     print("<debug>end_ids:",end_ids)

    #     print("<debug>zero_tokenizer:", self.tokenizer.convert_ids_to_tokens(0)) 

    #     def find_sublist_positions(response, sublist):
    #         """在 response 中找到 sublist 的起始位置"""
    #         n, m = len(response), len(sublist)
    #         for i in range(n - m + 1):
    #             if response[i:i+m] == sublist:
    #                 return i
    #         return -1

    #     print("<debug>response:",response)
    #     seq_list = response.tolist()
    #     print("<debug>sub_list:",seq_list)
    #     start_pos = find_sublist_positions(seq_list, start_ids)
    #     print("<debug>start_pos:",start_pos)
    #     end_pos   = find_sublist_positions(seq_list, end_ids)
    #     print("<debug>end_pos:",end_pos)

    #     if start_pos == -1:
    #         # 没找到标签，默认全保留
    #         mask = torch.ones_like(response)
    #         print("<debug>没有找到<result>")
    #         return response, mask, {"start_pos": None, "end_pos": None}

    #     # </Result> 最后一个token的位置
    #     end_pos_end = end_pos + len(end_ids) - 1

    #     # 构造mask,从"<Result>"开始全部mask掉
    #     mask = torch.ones_like(response)
    #     mask[start_pos:] = 0

    #     # masked_response = response * mask
    #     # 直接去start_pos之前的内容
    #     if start_pos is not None and start_pos > 0:
    #         masked_response = response[:start_pos] 

    #     return masked_response, mask, {"start_pos": start_pos, "end_pos_end": end_pos_end}
    
    # def mask_after_response(self, response, tokenizer, start_tag="<Result>", end_tag="</Result><|im_end|>"):
    #     """
    #     在序列中找到 <Result> ... </Result> 的范围，并把 </Result> 之后的内容全部mask掉
    #     """
    #     # decode 整个 response
    #     decoded_text = tokenizer.decode(response, skip_special_tokens=False)

    #     # 找到字符级别的起止位置
    #     start_char = decoded_text.find(start_tag)
    #     print("<start_char>:", start_char)
    #     end_char   = decoded_text.find(end_tag)
    #     print("<end_char>:", end_char)
    #     if start_char == -1 or end_char == -1:
    #         print("<debug>没有找到<Result>")
    #         mask = torch.ones_like(response)
    #         return response, mask, {"start_pos": None, "end_pos": None}

    #     # 用 tokenizer.batch_decode + offsets 映射回 token 级别位置
    #     enc = tokenizer(decoded_text, return_offsets_mapping=True, add_special_tokens=False)
    #     offsets = enc["offset_mapping"]

    #     # 找到对应的 token 范围
    #     start_pos, end_pos = None, None
    #     for i, (s, e) in enumerate(offsets):
    #         if s <= start_char < e and start_pos is None:
    #             start_pos = i
    #         if s < end_char < e:
    #             end_pos = i
    #             break

    #     print("<debug>start_pos:", start_pos)
    #     print("<debug>end_pos:", end_pos)

    #     if start_pos is None:
    #         mask = torch.ones_like(response)
    #         return response, mask, {"start_pos": None, "end_pos": None}

    #     # 构造 mask
    #     mask = torch.ones_like(response)
    #     mask[start_pos:]decode = 0

    #     masked_response = response[:start_pos]

    #     return masked_response, mask, {"start_pos": start_pos, "end_pos": end_pos}

    def mask_after_response(self, response, tokenizer,
                            start_tag="<Result>",
                            end_tag="</Result><|im_end|>",
                            debug: bool = True):
        """
        使用 字符串 + offset_mapping 的方式找到 <Result>...</Result><|im_end|> 的 token 范围，
        返回 (response, mask, info_dict)
        - mask: 与 response 长度相同的 1/0 向量 (1 表示保留, 0 表示被 mask)
        - info_dict 包含 start_pos, end_pos, content_text, debug 额外信息
        debug=True 时会打印 token/offset/覆盖情况
        """
        # 支持 response 为 torch.Tensor 或 list[int]
        if isinstance(response, torch.Tensor):
            ids = response.tolist()
        else:
            ids = list(response)

        # decode 得到纯文本（包含特殊符号）
        decoded_text = tokenizer.decode(ids, skip_special_tokens=False)

        # 找字符级别的标签起止
        start_char = decoded_text.find(start_tag)
        end_char = decoded_text.find(end_tag)
        if start_char == -1 or end_char == -1:
            if debug:
                print("[debug] 没找到 start_tag 或 end_tag:", start_tag, end_tag)
            # 兜底返回全1 mask（不屏蔽）
            mask = torch.ones(len(ids), dtype=torch.long) if not isinstance(response, torch.Tensor) else torch.ones_like(response)
            return response, mask, {"start_pos": None, "end_pos": None, "content": None, "reason": "tag_not_found"}

        start_char_end = start_char + len(start_tag)   # 标签结束的字符索引（exclusive）
        end_char_end = end_char + len(end_tag)

        # 用 tokenizer 对 decoded_text 编码以获得 offset_mapping（字符->token 映射）
        enc = tokenizer(decoded_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]    # list of (s,e)
        enc_ids = enc["input_ids"]         # token ids 对应 decoded_text
        tokens = tokenizer.convert_ids_to_tokens(enc_ids)

        if debug:
            print(f"[debug] decoded_text: {decoded_text!r}")
            print(f"[debug] start_char..start_char_end = [{start_char}, {start_char_end}), end_char..end_char_end = [{end_char}, {end_char_end})")
            print(f"[debug] tokens count: {len(tokens)}")
            for i, (tok, tid, (s, e)) in enumerate(zip(tokens, enc_ids, offsets)):
                mark = ""
                # 标注与 start_tag / end_tag 区间有交集的 token
                if not (e <= start_char or s >= start_char_end):
                    mark += " <-- overlaps start_tag"
                if not (e <= end_char or s >= end_char_end):
                    mark += " <-- overlaps end_tag"
                print(f"  token[{i:3d}] = {tok:12s} id={tid:6d} offset=({s},{e}){mark}")

        # 找到覆盖 start_tag 的 token 下标集合
        start_indices = [i for i, (s, e) in enumerate(offsets) if (s < start_char_end and e > start_char)]
        # 找到覆盖 end_tag 的 token 下标集合
        end_indices = [i for i, (s, e) in enumerate(offsets) if (s < end_char_end and e > end_char)]

        info = {"start_tag_token_indices": start_indices, "end_tag_token_indices": end_indices}

        content_start_pos = None
        content_end_pos = None

        if start_indices and end_indices:
            # 内容开始于 start_tag 覆盖 token 的最后一个 token 的下一位
            content_start_pos = start_indices[-1] + 1
            # 内容结束于 end_tag 覆盖 token 的第一个 token 的位置（exclusive）
            content_end_pos = end_indices[0]
        else:
            # 如果 offset 映射出现问题，尝试做 token-id 子序列匹配
            if debug:
                print("[debug] offset 映射未覆盖完整标签，开始使用 token-id 子序列匹配作为 fallback")
            start_ids = tokenizer(start_tag, add_special_tokens=False)["input_ids"]
            end_ids   = tokenizer(end_tag, add_special_tokens=False)["input_ids"]

            # 在 enc_ids 中查找 start_ids 子序列
            start_match = None
            for i in range(0, len(enc_ids) - len(start_ids) + 1):
                if enc_ids[i:i+len(start_ids)] == start_ids:
                    start_match = i + len(start_ids)  # 内容开始
                    break

            end_match = None
            if start_match is not None:
                for i in range(start_match, len(enc_ids) - len(end_ids) + 1):
                    if enc_ids[i:i+len(end_ids)] == end_ids:
                        end_match = i
                        break

            if start_match is not None and end_match is not None:
                content_start_pos = start_match
                content_end_pos = end_match
                info["fallback"] = "token_id_subsequence"
                info["start_ids"] = start_ids
                info["end_ids"] = end_ids
            else:
                # 仍然失败 -> 返回未找到
                if debug:
                    print("[debug] fallback 也失败，无法定位标签")
                mask = torch.ones(len(ids), dtype=torch.long) if not isinstance(response, torch.Tensor) else torch.ones_like(response)
                info["reason"] = "offset_and_fallback_failed"
                return response, mask, {"start_pos": None, "end_pos": None, "content": None, **info}

        # 边界检查（确保在合理范围内）
        seq_len = len(enc_ids)
        if content_start_pos < 0:
            content_start_pos = 0
        if content_end_pos is None:
            content_end_pos = seq_len
        if content_start_pos > seq_len:
            content_start_pos = seq_len
        if content_end_pos > seq_len:
            content_end_pos = seq_len

        # 用 token span 提取内容文本（debug / 返回用）
        content_token_ids = enc_ids[content_start_pos:content_end_pos]
        content_text = tokenizer.decode(content_token_ids, skip_special_tokens=False).strip()

        # 构造 mask：mask 从 content_end_pos 开始置 0（即 </Result> 及其之后会被保留或屏蔽，依据你原始逻辑）
        # 这里：mask[end_pos:] = 0  -> 把 end_pos 之后（含 end_pos) 全部置 0
        # end_pos 指向的是 end_tag 首个 token 的位置（content_end_pos）
        if isinstance(response, torch.Tensor):
            mask = torch.ones_like(response)
        else:
            mask = torch.ones(len(ids), dtype=torch.long)

        # mask 长度可能与 enc_ids 长度一致（通常一样），如果不一致需要适配（这里假定一致）
        try:
            mask[content_end_pos:] = 0
        except Exception as e:
            # 安全兜底：若索引出错（长度不同），返回全1
            if debug:
                print("[debug] mask indexing error:", e)
            mask = torch.ones(len(ids), dtype=torch.long) if not isinstance(response, torch.Tensor) else torch.ones_like(response)
            info["reason"] = "mask_index_error"
            return response, mask, {"start_pos": None, "end_pos": None, "content": None, **info}

        info.update({
            "start_pos": content_start_pos,
            "end_pos": content_end_pos,
            "content": content_text,
        })

        if debug:
            print(f"[debug] content token span = [{content_start_pos}, {content_end_pos})")
            print(f"[debug] extracted content: {content_text!r}")

        return response, mask, info

    # 分组计算置信度系数
    # def normalize_seq_confs(self, processed_items):
    #     # 按uid分组
    #     groups = defaultdict(list)
    #     for item in processed_items:
    #         groups[item["uid"]].append(item)
    #     print("<debug>groups:", groups)

    #     for uid, items in groups.items():
    #         # 再按answer_reward分小组
    #         subgroups = defaultdict(list)
    #         for item in items:
    #             subgroups[item["answer_reward"]].append(item)
    #         print("<debug>subgroups:", subgroups)

    #         for reward_val, subgroup in subgroups.items():
    #             # 剔除format_reward == -1的，用于归一化的候选集
    #             valid_items = [it for it in subgroup if it["format_reward"] != -1]
    #             print("<debug>valid_items:", valid_items)
    #             if valid_items:
    #                 confs = [it["seq_confs"] for it in valid_items]
    #                 print("<debug>confs:", confs)
    #                 min_conf, max_conf = min(confs), max(confs)
    #                 # 避免除0错误
    #                 if max_conf == min_conf:
    #                     for it in valid_items:
    #                         it["seq_confs_norm"] = 1  # 也就是说seq_confs都一样
    #                 else:
    #                     for it in valid_items:
    #                         print("<debug>it:", it)
    #                         # max-min归一化
    #                         it["seq_confs_norm"] = (it["seq_confs"] - min_conf) / (max_conf - min_conf)
    #                         print("<debug>seq_confs_norm:", it["seq_confs_norm"])

    #             # 对format_reward == -1的，直接设为1
    #             for it in subgroup:
    #                 if it["format_reward"] == -1:
    #                     it["seq_confs_norm"] = 1
        
    #     return processed_items
    
    # def normalize_seq_confs(self, processed_items):
    #     # 按uid分组
    #     groups = defaultdict(list)
    #     for item in processed_items:
    #         groups[item["uid"]].append(item)
    #     print("<debug>groups:", groups)

    #     for uid, items in groups.items():
    #         # 再按answer_reward分小组
    #         subgroups = defaultdict(list)
    #         for item in items:
    #             subgroups[item["answer_reward"]].append(item)
    #         print("<debug>subgroups:", subgroups)

    #         for reward_val, subgroup in subgroups.items():
    #             # 剔除format_reward == -1的，用于归一化的候选集
    #             valid_items = [it for it in subgroup if it["format_reward"] != -1]
    #             print("<debug>valid_items:", valid_items)
    #             if valid_items:
    #                 confs = [it["seq_confs"] for it in valid_items]
    #                 print("<debug>confs:", confs)
    #                 min_conf, max_conf = min(confs), max(confs)
    #                 # 避免除0错误
    #                 if max_conf == min_conf:
    #                     for it in valid_items:
    #                         it["seq_confs_norm"] = 1  # 也就是说seq_confs都一样
    #                 else:
    #                     for it in valid_items:
    #                         print("<debug>it:", it)
    #                         # max-min归一化
    #                         it["seq_confs_norm"] = (it["seq_confs"] - min_conf) / (max_conf - min_conf)
    #                         print("<debug>seq_confs_norm:", it["seq_confs_norm"])

    #         # 对format_reward == -1的，直接设为1
    #         for it in subgroup:
    #             if it["format_reward"] == -1:
    #                 it["seq_confs_norm"] = 1
        
    #     return processed_items
    
    # 分别算相似度
    # def normalize_seq_confs(self, processed_items):
    #     # 按uid分组
    #     groups = defaultdict(list)
    #     for item in processed_items:
    #         groups[item["uid"]].append(item)
    #     print("<debug>groups:", groups)

    #     for uid, items in groups.items():
    #         # 再按answer_reward分小组
    #         subgroups = defaultdict(list)
    #         for item in items:
    #             subgroups[item["answer_reward"]].append(item)
    #         print("<debug>subgroups:", subgroups)

    #         for reward_val, subgroup in subgroups.items():
    #             # 再剔除format_reward == -1
    #             # valid_items = [it for it in subgroup if it["format_reward"] != -1]
    #             valid_items = [
    #                 it for it in subgroup 
    #                 if it["format_reward"] != -1 
    #                 and it["analysis_content"] is not None 
    #                 and it["criterion_content"] is not None
    #             ]
    #             print("<debug>valid_items:", valid_items)
                
    #             # 只有一条文本内容直接赋值
    #             if valid_items:
    #                 if len(valid_items) == 1:
    #                     for it in valid_items:
    #                         it["analysis_sim"] = 1
    #                         it["criterion_sim"] = 1
    #                         it["valid_analysis_sim"] = 1
    #                         it["valid_criterion_sim"] = 1
    #                         print(f"<debug>单文本场景：sim都设为（文本数量：{len(valid_items)}）")
    #                 else:
    #                     # # seq_confs归一化
    #                     confs = [it["seq_confs"] for it in valid_items]
    #                     print("<debug>confs:", confs)
    #                     min_conf, max_conf = min(confs), max(confs)
    #                     # 避免除0错误
    #                     if max_conf == min_conf:
    #                         for it in valid_items:
    #                             it["seq_confs_norm"] = 1  # 所有seq_confs相同
    #                     else:
    #                         for it in valid_items:
    #                             print("<debug>it:", it)
    #                             # max-min归一化
    #                             it["seq_confs_norm"] = (it["seq_confs"] - min_conf) / (max_conf - min_conf)
    #                             print("<debug>seq_confs_norm:", it["seq_confs_norm"])
                        
    #                     # 处理analysis_content相似度
    #                     analysis_texts = [
    #                         str(it["analysis_content"]).strip() 
    #                         for it in valid_items 
    #                         if str(it["analysis_content"]).strip()
    #                     ]

    #                     print("<debug>analysis_texts count:", len(analysis_texts))  # 文本数量
    #                     print("<debug>analysis_texts ", analysis_texts)

    #                     start_embed_call = time.time()
                        
    #                     # 计算analysis相似度
    #                     analysis_sim = self.call_embed_service(analysis_texts)
    #                     print("<debug> analysis_sim:", analysis_sim)
    #                     analysis_sim_matrix = analysis_sim["similarity_matrix"]
    #                     print("<debug> analysis_sim_matrix:", analysis_sim_matrix) 

    #                     end_embed_call = time.time()
    #                     embed_call_total = end_embed_call - start_embed_call
    #                     print(f"<debug>analysis_sim_matrix: 单次embed服务调用总耗时 {embed_call_total:.4f} 秒（文本数量：{len(analysis_texts)}）")

    #                     print("<debug>analysis_sim_matrix is None:", analysis_sim_matrix is None)  # 校验矩阵


    #                     if analysis_texts and analysis_sim_matrix is not None:
    #                         if len(analysis_sim_matrix) == len(valid_items):
    #                             for idx, it in enumerate(valid_items):
    #                                 row_sum = sum(analysis_sim_matrix[idx])
    #                                 row_len = len(analysis_sim_matrix[idx])
    #                                 # 确保row_len > 1，避免除零
    #                                 it["analysis_sim"] = (row_sum - 1) / (row_len - 1) if row_len > 1 else 0.0
    #                                 print(f"<debug>item {idx} analysis_sim:", it["analysis_sim"])
    #                                 it["valid_analysis_sim"] = 1 #边缘情况，final_answer_reward是-2
    #                         else:
    #                             for it in valid_items:
    #                                 it["analysis_sim"] = 0.0
    #                                 print("<debug>矩阵长度不匹配，analysis_sim设为0")
    #                                 it["valid_analysis_sim"] = 0
    #                     else:
    #                         # 文本为空或矩阵为None，设默认值
    #                         for it in valid_items:
    #                             it["analysis_sim"] = 0
    #                             print("<debug>analysis_texts empty or matrix None, set to 0")
    #                             it["valid_analysis_sim"] = 0

    #                     # 计算criterion_content相似度
    #                     critisim_texts = [
    #                         str(it["criterion_content"]).strip() 
    #                         for it in valid_items 
    #                         if str(it["criterion_content"]).strip()
    #                     ]

    #                     print("<debug>critisim_texts count:", len(analysis_texts))  # 文本数量
    #                     print("<debug>critisim_texts ", analysis_texts)
    #                     start_embed_call = time.time()

    #                     critisim_sim = self.call_embed_service(critisim_texts)
    #                     print("<debug> critisim_texts:", critisim_texts)
    #                     critisim_sim_matrix = critisim_sim["similarity_matrix"]
    #                     print("<debug> critisim_sim_matrix:", critisim_sim_matrix)

    #                     end_embed_call = time.time()
    #                     embed_call_total = end_embed_call - start_embed_call
    #                     print(f"<debug>critisim_sim_matrix: 单次embed服务调用总耗时 {embed_call_total:.4f} 秒（文本数量：{len(critisim_texts)}）")

    #                     print("<debug>critisim_sim_matrix is None:", critisim_sim_matrix is None)

    #                     if critisim_texts and critisim_sim_matrix is not None:
    #                         if len(critisim_sim_matrix) == len(valid_items):
    #                             for idx, it in enumerate(valid_items):
    #                                 row_sum = sum(critisim_sim_matrix[idx])
    #                                 row_len = len(critisim_sim_matrix[idx])
    #                                 it["criterion_sim"] = (row_sum - 1) / (row_len - 1) if row_len > 1 else 0.0
    #                                 print(f"<debug>item {idx} criterion_sim:", it["criterion_sim"])
    #                                 it["valid_criterion_sim"] = 1 
    #                         else:
    #                             for it in valid_items:
    #                                 it["criterion_sim"] = 0.0
    #                                 print("<debug>矩阵长度不匹配，criterion_sim设为0")
    #                                 it["valid_criterion_sim"] = 0
    #                     else:
    #                         for it in valid_items:
    #                             it["criterion_sim"] = 0.0
    #                             print("<debug>文本为空或矩阵无效，criterion_sim设为0")
    #                             it["valid_criterion_sim"] = 0

    #             # 处理format_reward == -1的元素：sim字段设为0，seq_confs_norm设为1
    #             for it in subgroup:
    #                 if it["format_reward"] == -1 or it["criterion_content"] is None or it["analysis_content"] is None :
    #                     # 未满足结构直接final_reward = -3
    #                     it["seq_confs_norm"] = 1
    #                     it["analysis_sim"] = 0
    #                     it["criterion_sim"] = 0
    #                     it["valid_analysis_sim"] = -1
    #                     it["valid_criterion_sim"] = -1
    #                     print("<debug>format_reward=-1 item:", it)
            
    #     return processed_items

    # 计算response的相似度
    def normalize_seq_confs(self, processed_items):
        # 按 uid 分组
        groups = defaultdict(list)
        for item in processed_items:
            groups[item["uid"]].append(item)

        for uid, items in groups.items():
            # 再按 answer_reward 分组
            subgroups = defaultdict(list)
            for item in items:
                subgroups[item["answer_reward"]].append(item)

            for reward_val, subgroup in subgroups.items():
                # 分离有效和无效样本
                valid_items = [it for it in subgroup if it["format_reward"] != -1 and it["response_content"] is not None]
                invalid_items = [it for it in subgroup if it["format_reward"] == -1 or it["response_content"] is None]

                # 处理无效样本
                for it in invalid_items:
                    it["rank_level"] = None
                    it["valid_response_content_sim"] = -1
                    it["response_content_sim"] = 0

                # 没有有效样本直接跳过
                if not valid_items:
                    continue

                # 归一化 + 相似度计算（仅对有效样本）
                if len(valid_items) == 1:
                    # 单条样本直接设相似度为1
                    for it in valid_items:
                        it["response_content_sim"] = 1
                        it["valid_response_content_sim"] = 1
                        it["seq_confs_norm"] = 1
                        it["rank_level"] = None
                        print(f"<debug>单文本场景：sim都设为1（文本数量：{len(valid_items)}）")
                else:
                    # # seq_confs 归一化
                    # confs = [it["seq_confs"] for it in valid_items]
                    # min_conf, max_conf = min(confs), max(confs)
                    # for it in valid_items:
                    #     it["seq_confs_norm"] = 1 if max_conf == min_conf else (it["seq_confs"] - min_conf) / (max_conf - min_conf)

                    # 计算 response_content 相似度
                    response_texts = [str(it["response_content"]).strip() for it in valid_items]
                    print("response_texts", response_texts)
                    start_time = time.time()
                    resp_sim = self.call_embed_service(response_texts)
                    sim_matrix = resp_sim.get("similarity_matrix")
                    print("sim_matrix", sim_matrix)
                    print(f"<debug>embed耗时 {time.time() - start_time:.4f}s")

                    if sim_matrix and len(sim_matrix) == len(valid_items):
                        for idx, it in enumerate(valid_items):
                            row = sim_matrix[idx]
                            row_sum = sum(row)
                            row_len = len(row)
                            it["response_content_sim"] = (row_sum - 1) / (row_len - 1) if row_len > 1 else 0.0
                            print("response_content_sim" , it["response_content_sim"])
                            it["valid_response_content_sim"] = 1
                    else:
                        for it in valid_items:
                            it["response_content_sim"] = 0.0
                            it["valid_response_content_sim"] = 0

                # 排名逻辑（仅针对 answer_reward == 1）
                if valid_items:
                    if reward_val == 1:
                        # 按 response_content_sim 从高到低排序
                        sorted_items = sorted(valid_items, key=lambda x: x.get("response_content_sim", 0), reverse=True)
                        print("sorted_items:", sorted_items)
                        n = len(sorted_items)
                        print("n:", n)
                        split_index = math.ceil(n / 2)
                        print("split_index:", split_index)
                        for i, it in enumerate(sorted_items):
                            it["rank_level"] = "high" if i < split_index else "low"
                    else:
                        # 其他 reward_val 的情况，不分级
                        for it in valid_items:
                            it["rank_level"] = None

        return processed_items

    def __call__(self, data: DataProto, experiment_list: defaultdict, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # print("*" * 20)
        # print("raw data:",data)
        # print("*" * 20)

        processed_items = []

        uid_scores = defaultdict(list)
        rollout_score_list = defaultdict(list)
        added_sft_uids = set()
        uid_scores_update = defaultdict(list)
        added_rollout_mode_uids_i = set()
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            print("*" * 20)
            print("<debug>raw data_item:", data_item)
            print("*" * 20)
            uid = data_item.non_tensor_batch['uid']
            print("<debug>uid:", uid)

            # 新建一个变量用来区分同组内的数据
            uid_x = str(uid) + "_" + str(i)

            prompt_ids = data_item.batch['prompts']
            # print("<debug>prompt_ids:",prompt_ids)

            # sft_rollout score 加到对应uid的socre list里面，一个uid只加一次
            sft_score_list = data_item.non_tensor_batch['median_score_list']
            qid = data_item.non_tensor_batch['qid']
            expriment_score_list = experiment_list[qid]
            print(f"<debug>uid:{uid}, sft_score_list:{sft_score_list}")
            print(f"<debug>当前经验数组 uid:{uid}, expriment_score_list:{expriment_score_list}")
            if uid not in added_sft_uids and expriment_score_list is not None:
                if uid not in uid_scores:
                    uid_scores[uid] = []
                uid_scores[uid].extend(expriment_score_list)
                added_sft_uids.add(uid)

            # 拿到sft_rollout_mode众数
            uid_mid_score = data_item.non_tensor_batch["median_score"]
            # 众数
            # uid_mid_score = data_item.non_tensor_batch["mode_score"]
            print("<debug>uid_mid_score:", uid_mid_score)

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            # print("valid_prompt_ids:",valid_prompt_ids)'

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # print("valid_response_ids:",valid_response_ids)
            # print("<debug>valid_response_length:",valid_response_length)

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            response_str = self.tokenizer.decode(valid_response_ids)
            print("<debug>response_str:", response_str)
            sequences_str = self.tokenizer.decode(sequences)
            print("<debug>sequences_str:", sequences_str)
            
            # 检查格式
            is_valid_format, criterion_content, analysis_content = self.check_format(response_str)

            # 优先考虑格式
            if is_valid_format:
                # 提取当前response的token log_probs
                old_log_probs_list = data_item.batch["old_log_probs"]
                print("<debug>old_log_probs_list:", old_log_probs_list)
                
                # 获取<Result>对应token的首位
                response_cot, mask, result_prosition = self.mask_after_response(valid_response_ids, self.tokenizer)

                print("<debug>result_prosition:", result_prosition)
                # 解码除去<Result>后的内容
                response_cot_str = self.tokenizer.decode(response_cot)
                print("<debug>response_cot_str:", response_cot_str)

                # 提取token的首位
                start_pos = result_prosition["start_pos"]
                print("<debug>extra_start_pos:", start_pos)
                end_pos = result_prosition["end_pos"]
                print("<debug>extra_end_pos:", end_pos)
                mask = torch.ones_like(old_log_probs_list)
                mask[start_pos:] = 0

                # 得到掩码后对应的log_probs_list
                # real_log_probs_list = old_log_probs_list * mask
                # print("<debug>real_log_probs_list:", real_log_probs_list)

                # 直接取对应index之前的所有内容
                # if start_pos is not None and start_pos > 0:
                #     real_log_probs_list = old_log_probs_list[:start_pos]

                # 取对应result之前10%的内容
                # if start_pos is not None and start_pos > 0:
                #     valid_log_probs_list = old_log_probs_list[:start_pos]
                #     # 对于张量，使用numel()获取元素数量
                #     print("<debug>len of valid_log_probs_list:", valid_log_probs_list.numel())
                    
                #     # 检查张量是否包含元素（元素数量大于0）
                #     if valid_log_probs_list.numel() > 0:
                #         take_count = max(1, int(valid_log_probs_list.numel() * 0.1))
                #         print("<debug>take_count:", take_count)
                #         # 计算起始索引（确保不小于0）
                #         start_index = max(0, start_pos - take_count)
                #         print("<debug>start_index:", start_index)
                #         # 截取从start_index到start_pos的元素
                #         real_log_probs_list = old_log_probs_list[start_index:start_pos]
                #         print("<debug>real_log_probs_list:", real_log_probs_list)
                #     else:
                #         real_log_probs_list = torch.tensor([])  # 返回空张量保持类型一致
                # else:
                #     real_log_probs_list = torch.tensor([])
                
                # 取对应result之前64的内容
                if start_pos is not None and start_pos > 0:
                    valid_log_probs_list = old_log_probs_list[:start_pos]
                    # 对于张量，使用numel()获取元素数量
                    print("<debug>len of valid_log_probs_list:", valid_log_probs_list.numel())
                    
                    # 检查张量是否包含元素（元素数量大于0）
                    if valid_log_probs_list.numel() > 0:
                        take_count = min(64, valid_log_probs_list.numel())
                        print("<debug>take_count:", take_count)
                        # 截取标签前 take_count 个元素
                        real_log_probs_list = valid_log_probs_list[-take_count:]
                        print("<debug>real_log_probs_list:", real_log_probs_list)
                    else:
                        real_log_probs_list = torch.tensor([])  # 返回空张量保持类型一致
                else:
                    real_log_probs_list = torch.tensor([])

                # 取result之间的内容
                # if start_pos is not None and start_pos >= 0:
                #     if end_pos is not None and end_pos > start_pos:
                #         # 有效区间
                #         real_log_probs_list = old_log_probs_list[start_pos:end_pos]
                #         print("<debug>start_pos:", start_pos, "end_pos:", end_pos)
                #         print("<debug>real_log_probs_list:", real_log_probs_list)
                #         print("<debug>len of real_log_probs_list:", real_log_probs_list.numel())
                #     else:
                #         # 只有 start_pos，没有 end_pos（取到最后）
                #         real_log_probs_list = old_log_probs_list[start_pos:]
                #         print("<debug>start_pos:", start_pos, "end_pos:", end_pos)
                #         print("<debug>real_log_probs_list:", real_log_probs_list)
                #         print("<debug>len of real_log_probs_list:", real_log_probs_list.numel())
                # else:
                #     # 没找到 start_pos
                #     real_log_probs_list = torch.tensor([])
                #     print("<debug>start_pos:", start_pos, "end_pos:", end_pos)
                #     print("<debug>real_log_probs_list:", real_log_probs_list)
                #     print("<debug>len of real_log_probs_list:", real_log_probs_list.numel())

                # 取result前bottom 10%
                # if start_pos is not None and start_pos > 0:
                #     valid_log_probs_list = old_log_probs_list[:start_pos]
                #     print("<debug>len of valid_log_probs_list:", valid_log_probs_list.numel())

                #     # 检查张量是否包含元素（元素数量大于0）
                #     if valid_log_probs_list.numel() > 0:
                #         # 计算要取的数量（最少取1个）
                #         take_count = max(1, int(valid_log_probs_list.numel() * 0.1))
                #         print("<debug>take_count (10%):", take_count)

                #         # 获取最小的10%元素
                #         # torch.topk 默认是取最大的，这里用 largest=False 取最小的
                #         smallest_values, smallest_indices = torch.topk(valid_log_probs_list, k=take_count, largest=False)

                #         real_log_probs_list = smallest_values
                #         print("<debug>real_log_probs_list (bottom 10%):", real_log_probs_list)

                #     else:
                #         real_log_probs_list = torch.tensor([])  # 返回空张量保持类型一致
                # else:
                #     real_log_probs_list = torch.tensor([])

                # 核对是否正确
                # real_non_zero_count = sum(1 for val in real_log_probs_list if val != 0)
                # cot_non_zero_count = sum(1 for val in response_cot if val != 0)

                # 比较数量是否一致
                # if len(response_cot) == len(real_log_probs_list):
                #     print(f"<debug>result is mask")

                # 计算当前置信度
                seq_confs = self.compute_confidence(real_log_probs_list)
                print("<debug>seq_confs:", seq_confs)
                
                current_s = self.extract_score(response_str)
                print("<debug>current_s:", current_s)

                # 如果为空则不加入当前状态里
                if current_s is not None:
                    rollout_score_list[uid].append(current_s)

                processed_items.append({
                    'idx': i,
                    'uid': uid,
                    'uid_x': uid_x,
                    'sequences_str': sequences_str,
                    'current_s': current_s,
                    'is_valid_format': is_valid_format,
                    'valid_response_length': valid_response_length,
                    'response_content': response_str,
                    'uid_mid_score': uid_mid_score,
                    'expriment_score_list': expriment_score_list,
                    'rollout_score_list':rollout_score_list,
                    'sft_score_list': sft_score_list,
                    'start_pos': start_pos,
                    'seq_confs': seq_confs,
                    "criterion_content": criterion_content,
                    "analysis_content": analysis_content
                })
            else:
                ## 如果格式有误，那么这条数据的rollout_score_list是为空，从而不计算其对当前状态的影响
                processed_items.append({
                    'idx': i,
                    'uid': uid,
                    'uid_x': uid_x,
                    'sequences_str': sequences_str,
                    'current_s': None,
                    'is_valid_format': is_valid_format,
                    'valid_response_length': valid_response_length,
                    'response_content': response_str,
                    'uid_mid_score': uid_mid_score,
                    'sft_score_list': sft_score_list,
                    'start_pos': None,
                    'seq_confs': None,
                    "criterion_content": None,
                    "analysis_content": None
                })

        print("<debug>rollout_score_list:", rollout_score_list)
        print("<debug>uid_scores:", uid_scores)

        # 根据历史状态和当前的rollout结果计算当前状态下的gt
        baseline_processed_items = self.get_current_baseline(processed_items)

        current_baseline_map = self.build_uid_baseline_map(baseline_processed_items)
        print("<debug>current_baseline_map:", current_baseline_map)

        # 获取当轮状态下的众数
        uid_baseline = {
            uid: self.get_group_baseline(scores) 
            for uid, scores in rollout_score_list.items()
        }
        print("<debug>uid_baseline:", uid_baseline)

        # 更新经验数组
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            uid_i = data_item.non_tensor_batch['uid']
            print("<debug>uid_i:", uid_i)
            qid_i = data_item.non_tensor_batch['qid']
            print("<debug>qid_i:", qid_i)
            # 当前轮gt
            # current_baseline = uid_baseline.get(uid_i)
            # epoch_level + group_level -> current_baseline
            current_baseline = current_baseline_map.get(uid_i)
            print("<debug>current_baseline:", current_baseline)
            expriment_score_list = experiment_list[qid_i]
            print("<debug>expriment_score_list:", expriment_score_list)
            if isinstance(expriment_score_list, np.ndarray):
                # numpy数组转list
                expriment_score_list = expriment_score_list.tolist()
            # 控制同个uid下只添加一次current_baseline（rollout后的mode_score)
            if uid_i not in added_rollout_mode_uids_i and current_baseline is not None:
                # ## 先加新的数进去再算众数
                # expriment_score_list.append(current_baseline)
                # print("<debug>expriment_score_list:", expriment_score_list)
                # # 维护定长的经验数组，通过和众数的距离决定，初始是4
                # if len(expriment_score_list) > 6:
                #     mode = self.get_group_baseline(expriment_score_list)
                #     if mode is not None:
                #         distances = [abs(score - mode) for score in expriment_score_list]
                #         max_dist = max(distances)
                #         # 找到所有距离等于最大距离的索引
                #         candidate_indices = [i for i, d in enumerate(distances) if d == max_dist]
                #         # 随机选择一个索引移除
                #         index_to_remove = random.choice(candidate_indices)
                #     expriment_score_list.pop(index_to_remove)
                ## 先算众数剔除后再加新的数
                expected_length = len(expriment_score_list) + 1
                print("<debug>expriment_score_list:", expriment_score_list)
                # 维护定长的经验数组，通过和众数的距离决定，初始是8
                if expected_length > 20:
                    mode = self.get_group_baseline(expriment_score_list)
                    if mode is not None:
                        distances = [abs(score - mode) for score in expriment_score_list]
                        max_dist = max(distances)
                        # 找到所有距离等于最大距离的索引
                        candidate_indices = [i for i, d in enumerate(distances) if d == max_dist]
                        # 随机选择一个索引移除
                        index_to_remove = random.choice(candidate_indices)
                        expriment_score_list.pop(index_to_remove)
                        expriment_score_list.append(current_baseline)
                else:
                    expriment_score_list.append(current_baseline)
                # 更新过往经验
                experiment_list[qid_i] = expriment_score_list
                uid_scores_update[uid_i] = expriment_score_list
                added_rollout_mode_uids_i.add(uid_i)
            else:
                continue

        print("<debug>experiment_list_updated:", experiment_list)
        print("<debug>uid_scores_update:", uid_scores_update)

        # 根据过往经验和当前值算众数
        # uid_baseline_update = {
        #     uid: self.get_group_baseline(scores) 
        #     for uid, scores in uid_scores_update.items()
        # }
        # print("<debug>uid_baseline_update:", uid_baseline_update)


        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float16)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float16)
        answer_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float16)
        # gap_list = []

        # reward_tensor, format_reward_tensor, answer_reward_tensor = self.compute_reward_score(processed_items, reward, format_reward, answer_reward)
        
        # 计算format_reward和answer_reward
        for item in baseline_processed_items:
            i = item["idx"]
            uid = item['uid']
            print("<debug>uid:", uid)
            valid_response_length = item["valid_response_length"]
            print("*" * 20)
            print("<debug>当前处理数据:", item)
            print("*" * 20)
            print("<debug> response_content:", item["response_content"])
            # 格式奖励
            format_reward = 0.1 if item['is_valid_format'] else -1
            print("<debug> format_reward:", format_reward)
            # 分数奖励
            # baseline = uid_baseline.get(item['uid'])
            # print("uid_baseline:", baseline)
            # current_s = item['current_s']
            # print("current_score:", current_s)
            # if baseline is None or current_s is None:
            #     answer_reward = -1
            # else:
            #     answer_reward = 1 if current_s == baseline else -1

            # sft_rollout list
            print("<debug> sft_score_list:", item['sft_score_list'])
            
            # 获取当前状态下gt
            # 直接用组间中位数/众数作为gt
            # baseline_group = uid_baseline.get(item['uid'])
            # print("<debug> group_uid_baseline:", baseline_group)

            # 利用经验数组得到众数作为gt
            # baseline_group = uid_baseline_update.get(item['uid'])
            
            # 当前状态结合历史状态得到gt
            baseline_group = item['current_baseline']
            epoch_level_confidence = item['epoch_level_confidence']
            group_level_confidence = item['group_level_confidence']
            print("<debug> group_uid_baseline:", baseline_group)
            print("<debug> epoch_level_confidence:", epoch_level_confidence)
            print("<debug> group_level_confidence:", group_level_confidence)
            print(f"<debug> uid: {uid}, group_uid_baseline: {baseline_group}")

            # sft_rollout_gt值
            baseline = item['uid_mid_score']
            print("<debug> uid_baseline:", baseline)

            uid = item['uid']
            # print("<debug> score_value:", item['score_value'])
            current_s = item['current_s']
            print("<debug> current_s:", item['current_s'])
            # print("<debug> calculate_formula:", item['calculate_formula'])
            # print("<debug> verify_s:", item['verify_score'])
        
            # if baseline_group is None:
            #     logging.error(f"uid={uid}的baseline_group为None")
            #     answer_reward = -1
            if baseline_group is None:
                answer_reward = -1
                logging.warning(f"uid={uid}的baseline为None")

            elif current_s is None:
                answer_reward = -1
                logging.warning(f"uid={uid}的current_s为None")
            
            # elif start_pos is None:
            #     answer_reward = -1
            #     logging.warning(f"uid={uid}的start_pos为None")

            elif int(baseline_group) == 0:
                answer_reward = 0
                print("<debug> answer_reward = 0")

            else:
                # gap = baseline - baseline_group
                # print("<debug> baseline_gap:", gap)
                # current_s = round(item['current_s'])
                # print("<debug> current_s_after:", current_s)
                # distance = abs(current_s - baseline_group)
                # print("<debug> distance:", distance)

                if int(current_s) == int(baseline_group):
                    answer_reward = 1
                else:
                    answer_reward = -1
                # answer_reward = math.exp((1 - distance) / self.alpha) - 1
                #distance = 1的时候是临界值0
                
                # if distance == 0:
                #     answer_reward = 1
                # else:
                #     # answer_reward = -distance
                #     answer_reward = -1
                    
                print("<debug> answer_reward:", answer_reward)

            processed_items[i]['baseline_group'] = baseline_group
            processed_items[i]['format_reward'] = format_reward
            processed_items[i]['answer_reward'] = answer_reward

        # 结合token_level_confidence计算answer_reward，其中format_reward为-1的也不参与计算，直接-1
        final_processed_items = self.normalize_seq_confs(processed_items)
        
        for item in final_processed_items:
            i = item['idx']
            uid = item['uid']
            uid_x = item['uid_x']
            # seq_confs = item['seq_confs']
            # print("<debug>seq_confs:", seq_confs)

            current_s = item['current_s']
            print("<debug>current_s:", current_s)
            
            # analysis_sim = item['analysis_sim']
            # print("<debug>analysis_sim:", analysis_sim)
            # criterion_sim = item['criterion_sim']
            # print("<debug>criterion_sim:", criterion_sim)

            # valid_analysis_sim = item['valid_analysis_sim']
            # valid_criterion_sim = item['valid_criterion_sim']
            # print("<debug>valid_analysis_sim", valid_analysis_sim)
            # print("<debug>valid_criterion_sim", valid_criterion_sim)

            # 计算相似度系数 [Sim_p + (1- Sim_a)] / 2
            # sim_coef = (criterion_sim + (1 - analysis_sim) ) / 2
            # 计算相似度系数 alpha(1-simp) + (1-alpha)(1-sima)
            # alpha = 0.1
            # sim_coef = alpha * (1 - criterion_sim) + (1 - alpha) * (1 - analysis_sim)
            # print("sim_coef:", sim_coef)

            valid_response_length = item["valid_response_length"]
    
            format_reward = item['format_reward']
            print("<debug>format_reward:", format_reward)
            answer_reward = item['answer_reward']
            print("<debug>answer_reward:", answer_reward)
            # confs_coef = item['seq_confs_norm']
            # print("<debug>seq_confs_norm:", confs_coef)
            response_content_sim = item['response_content_sim']
            print("response_content_sim", response_content_sim)

            rank_level = item['rank_level']
            if rank_level == "high":
                rank_reward = 0.1
            else:
                rank_reward = 0

            # 计算answer奖励
            final_answer_reward = answer_reward + rank_reward
            print("<debug>final_answer_reward:", final_answer_reward)

            # 计算answer奖励
            # final_answer_reward = answer_reward * math.exp(confs_coef)
            # final_answer_reward = self.answer_weight * answer_reward * math.exp(sim_coef)
            # final_answer_reward = self.answer_weight * answer_reward * sim_coef

            # 处理极端情况
            if response_content_sim == 0:
                final_answer_reward = -2

            # final_answer_reward = answer_reward * confs_coef
            # if seq_confs != None:
            #     sigmoid_score = 1 / (1 + math.exp(seq_confs))
            # else:
            #     sigmoid_score = 1
            # print("<debug>sigmoid_score:", sigmoid_score)
            # final_answer_reward = answer_reward * sigmoid_score


            # 融合奖励
            final_reward = self.format_weight * format_reward + final_answer_reward
            # final_reward = self.format_weight * format_reward + self.answer_weight * answer_reward

            if format_reward == -1 or current_s == None:
                final_reward = -3 
            
            print("<debug>final_reward:", final_reward)
            print("*" * 20)

            # reward_tensor_before = reward_tensor.clone()

            reward_tensor[i, valid_response_length - 1] = final_reward
            format_reward_tensor[i, valid_response_length - 1]= format_reward
            answer_reward_tensor[i, valid_response_length - 1]= final_answer_reward

            # print(f"final_reward 值: {final_reward} (类型: {type(final_reward)})")
            # print(f"\n修改位置: [i={i}, valid_response_length-1={valid_response_length-1}]")

            # print("\n修改前的reward_tensor相关位置值:")
            # print(f"修改位置的值: {reward_tensor_before[i, valid_response_length - 1]}")
            # print(f"修改位置所在行的全部值: {reward_tensor_before[i]}")

            # print("\n修改后的reward_tensor相关位置值:")
            # print(f"修改位置的值: {reward_tensor[i, valid_response_length - 1]}")
            # print(f"修改位置所在行的全部值: {reward_tensor[i]}")

        reward_extra_info = {}
        reward_extra_info["answer_reward"] = reward_tensor
        reward_extra_info["format_reward_tensor"] = format_reward_tensor
        reward_extra_info["answer_reward_tensor"] = answer_reward_tensor
        # reward_extra_info[""]

        print(f"reward_tensor: {reward_tensor}, format_reward_tensor: {format_reward_tensor}, answer_reward_tensor: {answer_reward_tensor}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "experiment_list_update" : experiment_list
            }
        else:
            return reward_tensor, experiment_list

def trainer_dict_to_dataclass(conf: DictConfig):
    """Convert specific nested sections of a DictConfig object into dataclass instances.

    Args:
        conf (DictConfig): An instance of DictConfig, typically from the omegaconf library,
                           representing a configuration dictionary.

    Returns:
        DictConfig: A deep copy of the input `conf` with specific sections converted to dataclasses.
    """
    # Create a deep copy of the input configuration to avoid modifying the original object
    config = copy.deepcopy(conf)
    config.algorithm = omega_conf_to_dataclass(config.algorithm)
    config.critic.profiler = omega_conf_to_dataclass(config.critic.profiler)
    config.reward_model.profiler = omega_conf_to_dataclass(config.reward_model.profiler)
    config.actor_rollout_ref.actor.profiler = omega_conf_to_dataclass(config.actor_rollout_ref.actor.profiler)
    config.actor_rollout_ref.ref.profiler = omega_conf_to_dataclass(config.actor_rollout_ref.ref.profiler)
    config.actor_rollout_ref.rollout.profiler = omega_conf_to_dataclass(config.actor_rollout_ref.rollout.profiler)
    return config


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config_dict):
    config = trainer_dict_to_dataclass(config_dict)
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        # reward_fn = load_reward_manager(
        #     config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        # )
        # val_reward_fn = load_reward_manager(
        #     config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        # )
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
