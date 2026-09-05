//! 规则奖励 (Reward) 实现
//!
//! GRPO 闭环要求奖励是真实信号而非全 0 占位。这里实现两类规则：
//! 1. AnswerMatchReward: 从 response 中提取数值，与 instruction 中标注的答案比较
//! 2. FormatReward: 输出格式合理性 (长度、停止符、重复惩罚)
//!
//! 所有规则函数均为纯 CPU 实现，便于单元测试；`RewardModel` trait 提供 Tensor 包装以接入训练。

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// 从文本中提取第一个数字（支持负数、小数、`\boxed{...}`、中文"答案：X"格式）
pub fn extract_number(text: &str) -> Option<f64> {
    let text = text.trim();
    // \boxed{...} 优先
    if let Some(start) = text.find("boxed{") {
        if let Some(end_rel) = text[start + 6..].find('}') {
            let inner = text[start + 6..start + 6 + end_rel].trim();
            return parse_plain_number(inner);
        }
    }
    // 中文答案
    for marker in ["答案：", "答案:", "答案是", "answer is", "Answer:", "ANSWER:"] {
        if let Some(idx) = text.find(marker) {
            let tail = &text[idx + marker.len()..];
            if let Some(n) = parse_plain_number(tail) {
                return Some(n);
            }
        }
    }
    parse_plain_number(text)
}

fn parse_plain_number(s: &str) -> Option<f64> {
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let is_sign = (c == '-' || c == '+') && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
        let is_digit = c.is_ascii_digit()
            || (c == '.'
                && i + 1 < bytes.len()
                && bytes[i + 1].is_ascii_digit()
                && i > 0
                && bytes[i - 1].is_ascii_digit());
        if is_sign || is_digit {
            let start = i;
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == '.') {
                j += 1;
            }
            let cand: String = bytes[start..j].iter().collect();
            if let Ok(v) = cand.parse::<f64>() {
                // 拒绝形如 "1.2.3" 的错误串（上面已按首个数字串取）
                return Some(v);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    None
}

/// 判断 response 是否包含与 instruction 答案一致的数值
pub fn answer_match_score(instruction: &str, response: &str) -> f32 {
    match (extract_number(instruction), extract_number(response)) {
        (Some(expected), Some(got)) => {
            let diff = (expected - got).abs();
            let tol = 1e-4 * expected.abs().max(1.0);
            if diff <= tol {
                1.0
            } else {
                0.0
            }
        }
        (Some(_), None) => 0.0, // 指令要求答案但 response 没给数字
        (None, _) => 0.5,      // 指令没有明确答案，只能给中等分
    }
}

/// 格式奖励：长度适中给高分，过长/过短惩罚；以 `</s>` 等停止为佳
pub fn format_score(response: &str, min_len: usize, max_len: usize, eos_marker: &str) -> f32 {
    let len = response.chars().count();
    let mut score: f32 = 0.0;
    // 长度
    if len == 0 {
        return 0.0;
    } else if len < min_len {
        score += 0.2;
    } else if len <= max_len {
        score += 0.6;
    } else {
        score += 0.2; // 过长说明没学会停止
    }
    // 停止符
    if !eos_marker.is_empty() && response.contains(eos_marker) {
        score += 0.4;
    } else if eos_marker.is_empty() {
        score += 0.3;
    }
    score.min(1.0)
}

/// 组合规则奖励（默认权重）
pub fn rule_score(instruction: &str, response: &str, min_len: usize, max_len: usize) -> f32 {
    let ans = answer_match_score(instruction, response);
    // 指令带答案时以正确性为主；无答案时主要看格式
    if extract_number(instruction).is_some() {
        let fmt = format_score(response, min_len, max_len, "");
        0.8 * ans + 0.2 * fmt
    } else {
        let fmt = format_score(response, min_len, max_len, "");
        0.3 + 0.7 * fmt
    }
}

/// 奖励类型（CLI 可选）
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum RewardKind {
    /// 组合：数学答案匹配 + 格式
    Rule,
    /// 仅答案匹配
    Answer,
    /// 仅格式
    Format,
}

impl RewardKind {
    pub fn score(&self, instruction: &str, response: &str, min_len: usize, max_len: usize) -> f32 {
        match self {
            RewardKind::Rule => rule_score(instruction, response, min_len, max_len),
            RewardKind::Answer => answer_match_score(instruction, response),
            RewardKind::Format => format_score(response, min_len, max_len, ""),
        }
    }
}

/// 供训练在 CPU 端批量打分（字符串层面）
pub fn score_batch(
    kind: RewardKind,
    instructions: &[String],
    responses: &[Vec<String>],
    min_len: usize,
    max_len: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(responses.len() * responses[0].len());
    for (ins, group) in instructions.iter().zip(responses.iter()) {
        for r in group {
            out.push(kind.score(ins, r, min_len, max_len));
        }
    }
    out
}

// ===== Tensor 接口（向后兼容的 RewardModel trait） =====

/// 基础奖励函数接口
pub trait RewardModel<B: Backend> {
    /// 为一组生成的 Response 计算奖励值
    /// responses: Vec<String>
    /// 返回: [GroupSize] 的奖励 Tensor
    fn score(&self, instruction: &str, responses: &[String], device: &B::Device) -> Tensor<B, 1>;
}

/// 基于规则的奖励函数（数学验证 + 格式）
pub struct RuleReward {
    pub kind: RewardKind,
    pub min_len: usize,
    pub max_len: usize,
}

impl RuleReward {
    pub fn new() -> Self {
        Self {
            kind: RewardKind::Rule,
            min_len: 4,
            max_len: 512,
        }
    }
}

impl Default for RuleReward {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> RewardModel<B> for RuleReward {
    fn score(&self, instruction: &str, responses: &[String], device: &B::Device) -> Tensor<B, 1> {
        let scores: Vec<f32> = responses
            .iter()
            .map(|resp| self.kind.score(instruction, resp, self.min_len, self.max_len))
            .collect();
        Tensor::from_floats(scores.as_slice(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("答案是 5"), Some(5.0));
        assert_eq!(extract_number("答案: 5"), Some(5.0));
        assert_eq!(extract_number("答案：5"), Some(5.0));
        assert_eq!(extract_number("\\boxed{12.5}"), Some(12.5));
        assert_eq!(extract_number("结果是 -3"), Some(-3.0));
        assert_eq!(extract_number("Answer is 42"), Some(42.0));
        assert_eq!(extract_number("没有数字"), None);
    }

    #[test]
    fn test_answer_match() {
        assert_eq!(
            answer_match_score("计算 2+3，答案：5", "结果是 5"),
            1.0
        );
        assert_eq!(
            answer_match_score("计算 2+3，答案：5", "结果是 6"),
            0.0
        );
        assert_eq!(
            answer_match_score("计算 2+3，答案：5", "没有给出数字"),
            0.0
        );
    }

    #[test]
    fn test_format() {
        // 空回复 = 0
        assert_eq!(format_score("", 4, 64, ""), 0.0);
        // 合适长度
        let s = "第一步计算 2+3=5，因此答案是 5。";
        let f = format_score(s, 4, 64, "");
        assert!(f > 0.5, "合理长度格式分应较高, got {f}");
    }

    #[test]
    fn test_rule_score_bounds() {
        // 分数必须落在 [0,1]
        let cases = [
            ("计算 1+1，答案：2", "答案是 2"),
            ("写一段话", "今天天气很好，我们去公园散步，看到了很多花。"),
            ("", ""),
        ];
        for (ins, resp) in cases {
            let s = rule_score(ins, resp, 4, 64);
            assert!((0.0..=1.0).contains(&s), "score {s} out of range");
        }
    }
}
