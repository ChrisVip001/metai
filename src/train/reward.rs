use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// 基础奖励函数接口
pub trait RewardModel<B: Backend> {
    /// 为一组生成的 Response 计算奖励值
    /// responses: Vec<String>
    /// 返回: [GroupSize] 的奖励 Tensor
    fn score(&self, instruction: &str, responses: &[String], device: &B::Device) -> Tensor<B, 1>;
}

/// 基于规则的奖励函数示例 (如数学题目验证)
pub struct RuleReward;

impl<B: Backend> RewardModel<B> for RuleReward {
    fn score(&self, _instruction: &str, responses: &[String], device: &B::Device) -> Tensor<B, 1> {
        let scores: Vec<f32> = responses
            .iter()
            .map(|resp| {
                // 简单的评分逻辑: 如果包含 "answer" 关键词且长度适中，给高分
                if resp.contains("answer") && resp.len() > 10 {
                    1.0
                } else if resp.len() > 0 {
                    0.1
                } else {
                    0.0
                }
            })
            .collect();

        Tensor::from_floats(scores.as_slice(), device)
    }
}
