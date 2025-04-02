use std::io::{self, Write};
use std::path::Path;
use name2gender::{train_test_split, Name2gender};

fn main() {
    let model_path = Path::new("model.msgpack");

    let model = if model_path.exists() {
        println!("📦 Loading model from file...");
        Name2gender::load_from_file(model_path)
    } else {
        println!("🧠 Training new model...");
        let data = Name2gender::from_csv(Path::new("data/gender_type.csv"));
        let male_split = train_test_split(&data.male, 0.2);
        let female_split = train_test_split(&data.female, 0.2);

        let model = Name2gender::train_from_records(&male_split.train, &female_split.train);

        println!("💾 Saving model...");
        model.save_to_file(model_path);
        model
    };

    let train_acc = model.evaluate_on(&model.male, &model.female);
    println!("✅ Accuracy: {:.2}%", train_acc * 100.0);
    model.show_top_features(10);

    loop {
        print!("Enter a name to classify (or 'exit'): ");
        io::stdout().flush().unwrap();
        let mut name = String::new();
        io::stdin().read_line(&mut name).unwrap();
        let name = name.trim();

        if name.eq_ignore_ascii_case("exit") {
            break;
        }

        let (label, p_male, p_female) = model.predict_with_proba(name);
        println!(
            "{} is classified as {} (P_male = {:.2}%, P_female = {:.2}%)",
            name,
            label,
            p_male * 100.0,
            p_female * 100.0
        );
    }
}
