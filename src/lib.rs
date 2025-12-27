pub mod data;
pub mod infer;
pub mod model;
pub mod train;

// 重新导出常用类型
pub use infer::Generator;
pub use model::{MetaIConfig, MetaIModel};
pub use data::MetaITokenizer;
