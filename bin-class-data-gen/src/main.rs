#![allow(unused)]

use clap::Parser;
use tqdm::tqdm;
use rand::Rng;

/// Program to generate data for the bianry classifier
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// Maximum length of a sequence
    #[arg(short, long, default_value_t = 10)]
    max_length: i64,

    /// Group size to use
    #[arg(short, long, default_value_t = 5)]
    group_size: i64,

    /// Dataset size
    #[arg(short, long, default_value_t = 100_000)]
    dataset_size: i64,

    /// Identity proportion
    #[arg(short, long, default_value_t = 0.5)]
    identity_proportion: f64,

    /// Filename to write to
    #[arg(short, long, default_value_t = String::from("data.csv"))]
    filename: String,
}

fn is_identity(seq: &[i64], args: &Args) -> bool {
    // initialize the permutations
    let mut perm: Vec<i64> = (0..args.group_size).collect();
    
    // do the swaps
    for i in seq.iter() {
        // 0 is the identity
        // so we ignore it if we see it
        if *i != 0 {
            perm.swap(*i as usize, *i as usize + 1);
        }
    }

    // check if the permutation is the identity
    println!("perm {:?}", perm);
    return perm == (0..args.group_size).collect::<Vec<i64>>();
}

fn generate_random_sequnce(args: &Args) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    return (0..10).map(|_| rng.gen_range(0..args.group_size)).collect();
}

fn main() {
    let args = Args::parse();

    // print a random sequence
    println!("Random sequence: {:?}", generate_random_sequnce(&args));
}