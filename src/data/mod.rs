use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct MetaITokenizer {
    tokenizer: Tokenizer,
}

impl MetaITokenizer {
    pub fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(anyhow::Error::msg)?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.tokenizer.encode(text, true).expect("Encoding failed");
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).expect("Decoding failed")
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn train<P: AsRef<Path>>(
        files: &[P],
        output_path: P,
        vocab_size: usize,
    ) -> anyhow::Result<Self> {
        use tokenizers::decoders::DecoderWrapper;
        use tokenizers::models::bpe::{BpeTrainer, BPE};
        use tokenizers::normalizers::NormalizerWrapper;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::PreTokenizerWrapper;
        use tokenizers::processors::PostProcessorWrapper;
        use tokenizers::{AddedToken, TokenizerBuilder};

        let mut trainer = BpeTrainer::builder()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from("<pad>".to_string(), true),
                AddedToken::from("<s>".to_string(), true),
                AddedToken::from("</s>".to_string(), true),
                AddedToken::from("<unk>".to_string(), true),
            ])
            .build();

        // 显式提供所有包装器类型以满足 0.22.2 的泛型要求
        let mut tokenizer = TokenizerBuilder::<
            _,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())))
        .build()
        .map_err(anyhow::Error::msg)?;

        tokenizer
            .train_from_files(
                &mut trainer,
                files
                    .iter()
                    .map(|p| p.as_ref().to_string_lossy().into_owned())
                    .collect(),
            )
            .map_err(anyhow::Error::msg)?;

        tokenizer
            .save(output_path, true)
            .map_err(anyhow::Error::msg)?;

        Ok(Self {
            tokenizer: tokenizer.into(),
        })
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("<pad>")
    }
}

pub mod data;
pub mod dpo;
pub mod sft;
