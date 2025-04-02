use criterion::{criterion_group, criterion_main, Criterion};
use name2gender::Name2gender;
use std::path::Path;

fn bench_predict_single(c: &mut Criterion) {
    let model = Name2gender::load_from_file(Path::new("model.msgpack"));

    c.bench_function("predict Samantha", |b| {
        b.iter(|| {
            let _ = model.predict_with_proba("Samantha");
        })
    });
}

fn bench_bulk_prediction(c: &mut Criterion) {
    let model = Name2gender::load_from_file(Path::new("model.msgpack"));
    let names: Vec<_> = model
        .male
        .iter()
        .chain(model.female.iter())
        .take(10_000)
        .map(|r| r.name.clone())
        .collect();

    c.bench_function("bulk predict 10k names", |b| {
        b.iter(|| {
            for name in &names {
                let _ = model.predict_with_proba(name);
            }
        });
    });
}

criterion_group!(benches, bench_predict_single, bench_bulk_prediction);
criterion_main!(benches);
