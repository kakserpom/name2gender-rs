# name2gender 🧠🚻

A blazingly-fast, interpretable **gender classifier for first names**, powered by Rust 🦀 and [linfa](https://github.com/rust-ml/linfa)'s Naive Bayes implementation.

Inspired by the [jitsm555/Gender-Predictor](https://github.com/jitsm555/Gender-Predictor) Python project, this crate replicates and extends the idea using static typing, performance, and safe concurrency from Rust.

---

## ✨ Features

- 🔤 **Feature extraction** from name suffixes, prefixes, letters (e.g. `last3=ina`, `has=a`)
- 🧠 **Multinomial Naive Nayes classifier** (using `linfa-bayes`)
- 🎯 **Label + probability prediction**
- 💾 **Model caching** via [rmp-serde](https://crates.io/crates/rmp-serde)
- 🔁 **Auto-retrain** when the CSV is newer than the saved model
- 📈 **Benchmark-ready** with [criterion.rs](https://crates.io/crates/criterion)
- 🧪 **Train/test split evaluation**
- 🔍 **Most informative features viewer** 

---

## 🚀 Quickstart

### 1. Prepare your data

Your CSV file should look like this:

```csv
name,male_count,female_count
James,1050,10
Emily,5,1080
Skyler,300,320
```

Save as: `data/gender_type.csv`

---

### 2. Run the CLI app

```bash
cargo run
```

- Trains the model on first launch
- Saves to `model.msgpack`
- Auto-loads model on subsequent runs
- Re-trains only if CSV is modified

---

### 3. Sample output

```
🧠 Training new model...
💾 Saving model...
✅ Accuracy: 85.21%
Most Informative Features (based on freq ratio):
                last3=ena | male:    1, female:  545, ratio(f/m): 273.00
                last3=ina | male:    3, female:  814, ratio(f/m): 203.75
                last3=cia | male:    1, female:  365, ratio(f/m): 183.00
                last3=sia | male:    2, female:  475, ratio(f/m): 158.67
                last3=nda | male:    3, female:  610, ratio(f/m): 152.75
                last3=ica | male:    1, female:  272, ratio(f/m): 136.50
                last3=lla | male:    3, female:  532, ratio(f/m): 133.25
                last3=yla | male:    1, female:  257, ratio(f/m): 129.00
                last3=sha | male:   10, female: 1358, ratio(f/m): 123.55
                last3=isa | male:    1, female:  240, ratio(f/m): 120.50
```

---

## 🧠 Example usage

```rust
use name2gender::Name2gender;
use std::path::Path;

let model = Name2gender::load_or_train_if_stale(
    Path::new("model.msgpack"),
    Path::new("data/gender_type.csv"),
    0.2
);

let (label, p_male, p_female) = model.predict_with_proba("Herbert");
println!("Herbert → {label} ({:.2}%, {:.2}%)", p_male * 100.0, p_female * 100.0);
```

---

## 🧪 Benchmarking

```bash
cargo bench
```

Implemented via [criterion](https://crates.io/crates/criterion), tests both single-name and bulk classification.

---

## 🔧 Dependencies

| Crate         | Use                                      |
|---------------|-------------------------------------------|
| linfa         | Core ML framework                         |
| linfa-bayes   | Naive Bayes classifier                    |
| ndarray       | Numerical arrays                          |
| rmp-serde     | MessagePack serialization                 |
| serde         | Serialization of model and features       |
| csv           | Input data format                         |
| rand          | Data shuffling                            |
| criterion     | Benchmarking                              |

---

## 📚 Related Projects

- 🐍 Python: [jitsm555/Gender-Predictor](https://github.com/jitsm555/Gender-Predictor)
- 📊 Concept: [`scikit-learn` MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- 🦀 Rust ML: [linfa](https://github.com/rust-ml/linfa)

---

## 📄 License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

---

## 💬 Contributing

PRs, issues, and ideas are welcome!