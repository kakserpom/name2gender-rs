//! # name2gender 🧠🚻
//!
//! Predict gender from first names using a Naive Bayes classifier.
//!
//! Inspired by [jitsm555/Gender-Predictor](https://github.com/jitsm555/Gender-Predictor), this Rust crate extracts
//! character-level features (e.g. suffixes like `last3=ina`) from names and uses a [`linfa-bayes`](https://crates.io/crates/linfa-bayes)
//! Naive Bayes classifier to predict gender labels along with probability scores.
//!
//! ## Features
//! - N-gram and char-based feature extraction
//! - Multinomial Naive Bayes classifier
//! - Label + probability prediction
//! - Model persistence with `rmp-serde` (MessagePack)
//! - Auto-retrain when CSV is updated
//! - Benchmarkable with [Criterion](https://crates.io/crates/criterion)
//!
//! ## Example
//! ```rust
//! use std::path::Path;
//! use name2gender::Name2gender;
//! let model = Name2gender::load_or_train_if_stale(
//!     Path::new("model.msgpack"),
//!     Path::new("data/gender_type.csv"),
//!     0.2
//! );
//! let (label, p_male, p_female) = model.predict_with_proba("Samantha");
//! println!("Gender: {label}, P_male: {p_male:.2}, P_female: {p_female:.2}");
//! ```
#![warn(clippy::pedantic)]

use linfa::prelude::*;
use linfa_bayes::MultinomialNb;
use linfa_bayes::NaiveBayes;
use ndarray::{Array1, Array2};
use rand::rng;
use rand::seq::SliceRandom;
use rmp_serde::{decode::from_read, encode::write_named};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering::Equal;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::SystemTime;

/// A record representing a single name and its gender counts.
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct NameRecord {
    pub name: String,
    pub male_count: u32,
    pub female_count: u32,
}

/// A helper type for holding train/test splits.
#[derive(Debug)]
pub struct DatasetSplit {
    pub train: Vec<NameRecord>,
    pub test: Vec<NameRecord>,
}

/// Randomly splits a dataset into train and test sets based on `test_ratio`.
pub fn train_test_split(data: &[NameRecord], test_ratio: f64) -> DatasetSplit {
    let mut rng = rng();
    let mut data = data.to_vec();
    data.shuffle(&mut rng);

    let test_size = ((data.len() as f64) * test_ratio).round() as usize;
    let test = data[..test_size].to_vec();
    let train = data[test_size..].to_vec();

    DatasetSplit { train, test }
}

fn ngrams(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return vec![];
    }
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

fn extract_string_features(name: &str) -> BTreeSet<String> {
    let name = name.to_lowercase();
    let mut features = BTreeSet::new();

    let ngrams = ngrams(&name, 3);

    if let Some(first) = ngrams.first() {
        features.insert(format!("f={first}"));
    }
    if let Some(last) = ngrams.last() {
        features.insert(format!("l={last}"));
    }
    for c in ngrams {
        features.insert(format!("has={c}"));
    }
    features.insert(format!("is={name}"));

    features
}

/// Trained name-to-gender classifier and feature data.
#[derive(Serialize, Deserialize)]
pub struct Name2gender {
    model: MultinomialNb<f64, usize>,
    pub male: Vec<NameRecord>,
    pub female: Vec<NameRecord>,
    vocab: BTreeMap<String, usize>,
    feat_freq_male: HashMap<String, usize>,
    feat_freq_female: HashMap<String, usize>,
}

const LABEL_MALE: &str = "M";
const LABEL_FEMALE: &str = "F";

impl Name2gender {
    /// Load a saved model if up-to-date, or retrain if the CSV is newer.
    pub fn load_or_train_if_stale(
        model_path: &Path,
        csv_path: &Path,
        test_ratio: f64,
    ) -> anyhow::Result<Self> {
        let model_mtime = model_path
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let csv_mtime = csv_path
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let should_retrain = !model_path.exists() || csv_mtime > model_mtime;

        if should_retrain {
            println!("🧠 Training model (CSV is newer or model missing)...");
            let data = Name2gender::from_csv(csv_path)?;

            let male_split = train_test_split(&data.male, test_ratio);
            let female_split = train_test_split(&data.female, test_ratio);

            let model = Name2gender::train_from_records(&male_split.train, &female_split.train)?;
            println!("💾 Saving model to {model_path:?}");
            model.save_to_file(model_path)?;
            Ok(model)
        } else {
            println!("📦 Loading model from {model_path:?} (up-to-date)");
            Ok(Name2gender::load_from_file(model_path)?)
        }
    }

    /// Loads name data from CSV and constructs training samples.
    #[must_use]
    pub fn from_csv(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mut rdr = csv::Reader::from_reader(file);

        let mut male = vec![];
        let mut female = vec![];

        for result in rdr.deserialize() {
            let record: NameRecord = result?;

            match record.male_count.cmp(&record.female_count) {
                std::cmp::Ordering::Greater => {
                    male.push(record);
                }
                std::cmp::Ordering::Less => {
                    female.push(record);
                }
                std::cmp::Ordering::Equal => {}
            }
        }

        Ok(Name2gender {
            model: MultinomialNb::params()
                .fit(&Dataset::new(Array2::zeros((1, 1)), Array1::zeros(1)))?,
            male,
            female,
            vocab: BTreeMap::new(),
            feat_freq_male: HashMap::new(),
            feat_freq_female: HashMap::new(),
        })
    }

    /// Trains the classifier from given name records.
    pub fn train_from_records(male: &[NameRecord], female: &[NameRecord]) -> anyhow::Result<Self> {
        let mut all_features = vec![];
        let mut feat_freq_male = HashMap::new();
        let mut feat_freq_female = HashMap::new();

        for r in male {
            let feats = extract_string_features(&r.name);
            for f in &feats {
                *feat_freq_male.entry(f.clone()).or_insert(0) += 1;
            }
            all_features.push((0, feats));
        }

        for r in female {
            let feats = extract_string_features(&r.name);
            for f in &feats {
                *feat_freq_female.entry(f.clone()).or_insert(0) += 1;
            }
            all_features.push((1, feats));
        }

        let mut vocab = BTreeMap::new();
        let mut idx = 0;
        for (_, feats) in &all_features {
            for f in feats {
                if !vocab.contains_key(f) {
                    vocab.insert(f.clone(), idx);
                    idx += 1;
                }
            }
        }

        let mut feature_vecs = Vec::new();
        let mut labels = Vec::new();

        for (label, feats) in &all_features {
            let mut row = vec![0.0; vocab.len()];
            for f in feats {
                if let Some(&i) = vocab.get(f) {
                    row[i] = 1.0;
                }
            }
            feature_vecs.push(row);
            labels.push(*label);
        }

        let x = Array2::from_shape_vec((feature_vecs.len(), vocab.len()), feature_vecs.concat())?;
        let y = Array1::from_vec(labels);
        let dataset = Dataset::new(x, y);

        let model = MultinomialNb::params().fit(&dataset)?;

        Ok(Name2gender {
            model,
            male: male.to_vec(),
            female: female.to_vec(),
            vocab,
            feat_freq_male,
            feat_freq_female,
        })
    }

    /// Predicts gender and returns label (`"M"` or `"F"`) and class probabilities.
    #[must_use]
    pub fn predict_with_proba(&self, name: &str) -> anyhow::Result<(&'static str, f64, f64)> {
        let feats = extract_string_features(name);
        let mut row = vec![0.0; self.vocab.len()];

        for f in feats {
            if let Some(&i) = self.vocab.get(&f) {
                row[i] = 1.0;
            }
        }

        let input = Array2::from_shape_vec((1, self.vocab.len()), row)?;
        let (proba, classes) = self.model.predict_proba(input.view());

        let p_male = proba[[0, classes.iter().position(|&&c| c == 0).unwrap()]];
        let p_female = proba[[0, classes.iter().position(|&&c| c == 1).unwrap()]];

        let label = if p_male >= p_female {
            LABEL_MALE
        } else {
            LABEL_FEMALE
        };
        Ok((label, p_male, p_female))
    }

    /// Calculates classification accuracy on provided male/female datasets.
    #[must_use]
    pub fn evaluate_on(&self, male: &[NameRecord], female: &[NameRecord]) -> anyhow::Result<f64> {
        let mut correct = 0;
        let mut total = 0;

        for r in male {
            if self.predict_with_proba(&r.name)?.0 == LABEL_MALE {
                correct += 1;
            }
            total += 1;
        }

        for r in female {
            if self.predict_with_proba(&r.name)?.0 == LABEL_FEMALE {
                correct += 1;
            }
            total += 1;
        }

        Ok(f64::from(correct) / f64::from(total))
    }

    /// Displays most informative features based on female/male ratio.
    pub fn show_top_features(&self, n: usize) {
        println!("Most Informative Features (based on freq ratio):");

        let mut features: Vec<_> = self
            .vocab
            .keys()
            .map(|feat| {
                let male_freq = *self.feat_freq_male.get(feat).unwrap_or(&1);
                let female_freq = *self.feat_freq_female.get(feat).unwrap_or(&1);
                let ratio = (female_freq as f64 + 1.0) / (male_freq as f64 + 1.0);
                (feat.clone(), male_freq, female_freq, ratio)
            })
            .collect();

        features.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(Equal));

        for (feat, m, f, ratio) in features.iter().take(n) {
            println!("{feat:>25} | male: {m:>4}, female: {f:>4}, ratio(f/m): {ratio:.2}",);
        }
    }

    /// Saves the model to a binary `.msgpack` file.
    pub fn save_to_file(&self, path: &Path) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_named(&mut writer, self)?;
        Ok(())
    }

    /// Loads the model from a binary `.msgpack` file.
    pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(from_read(reader)?)
    }
}
