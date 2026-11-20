use std::path::Path;
use tokenizers::Tokenizer;

pub fn chat_prompt(instruction: &str, input: &str) -> String {
    if input.is_empty() {
        format!("<|user|>\n{instruction}\n<|assistant|>\n")
    } else {
        format!("<|user|>\n{instruction}\n{input}\n<|assistant|>\n")
    }
}

#[derive(Clone)]
pub struct MetaITokenizer {
    tokenizer: Tokenizer,
}

impl MetaITokenizer {
    pub fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(path).map_err(anyhow::Error::msg)?;
        // Historical 16K checkpoints have a ByteLevel pre-tokenizer but no matching decoder.
        if tokenizer.get_decoder().is_none() {
            if let Some(tokenizers::pre_tokenizers::PreTokenizerWrapper::ByteLevel(codec)) =
                tokenizer.get_pre_tokenizer().cloned()
            {
                tokenizer.with_decoder(Some(codec));
            }
        }
        Ok(Self { tokenizer })
    }

    pub fn equivalent(&self, other: &Self) -> anyhow::Result<bool> {
        Ok(serde_json::to_value(&self.tokenizer)? == serde_json::to_value(&other.tokenizer)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.tokenizer.save(path, true).map_err(anyhow::Error::msg)
    }

    pub fn response(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode(text);
        ids.extend(self.eos_id());
        ids
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.tokenizer.encode(text, true).expect("Encoding failed");
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).expect("Decoding failed")
    }

    /// Decode stored text verbatim, including literal strings matching reserved tokens.
    pub fn decode_verbatim(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).expect("Decoding failed")
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

        anyhow::ensure!(
            vocab_size >= 260,
            "byte-level BPE requires 256 bytes and four reserved tokens"
        );
        let codec = ByteLevel::default().add_prefix_space(false);
        let mut trainer = BpeTrainer::builder()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .special_tokens(vec![
                AddedToken::from("<pad>".to_string(), true),
                AddedToken::from("<s>".to_string(), true),
                AddedToken::from("</s>".to_string(), true),
                AddedToken::from("<unk>".to_string(), true),
            ])
            .build();

        // 显式提供所有包装器类型以满足 tokenizers 的泛型要求
        let mut tokenizer = TokenizerBuilder::<
            _,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >::new()
        .with_model(BPE::default())
        .with_decoder(Some(DecoderWrapper::ByteLevel(codec)))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(codec)))
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

    pub fn eos_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("</s>")
    }
}

#[allow(clippy::module_inception)] // Preserve the existing public module path.
pub mod data;
pub mod dpo;
pub mod sft;

pub use self::sft::{GRPODataset, GRPOItem};

#[cfg(test)]
mod compatibility_tests {
    use super::*;

    #[test]
    fn shipped_and_new_tokenizers_have_lossless_byte_fallback() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("train");
        let file = dir.path().join("trained.json");
        std::fs::write(&input, "only ascii training text")?;
        let trained = MetaITokenizer::train(&[&input], &file, 260)?;
        let shipped =
            MetaITokenizer::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json"))?;
        for codec in [shipped, trained] {
            for c in tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet() {
                assert!(codec.tokenizer.token_to_id(&c.to_string()).is_some());
            }
            for text in [
                "",
                "  空格\tand英文\r\n",
                "0123456789#%+/=@\\{}青橙47",
                "MAPLE-731🙂𠮷",
                "e\u{0301} / é",
                "<|user|>\n<|assistant|>",
            ] {
                assert_eq!(codec.decode(&codec.encode(text)), text);
            }
            let literal = "a<pad><s></s><unk>b";
            assert_eq!(codec.decode_verbatim(&codec.encode(literal)), literal);
        }
        Ok(())
    }

    #[test]
    fn legacy_bytelevel_decoder_preserves_ids_and_roundtrips_utf8() -> anyhow::Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json");
        let mut legacy: serde_json::Value = serde_json::from_reader(std::fs::File::open(&path)?)?;
        legacy["decoder"] = serde_json::Value::Null;
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("legacy.json");
        std::fs::write(&file, serde_json::to_vec(&legacy)?)?;
        let original = Tokenizer::from_file(&file).map_err(anyhow::Error::msg)?;
        let loaded = MetaITokenizer::new(&file)?;
        assert!(loaded.equivalent(&MetaITokenizer::new(path)?)?);
        let text = " The sky 蓝色";
        let ids = loaded.encode(text);
        assert_eq!(
            ids,
            original
                .encode(text, true)
                .map_err(anyhow::Error::msg)?
                .get_ids()
        );
        assert_eq!(loaded.decode(&ids), text);
        Ok(())
    }
}
