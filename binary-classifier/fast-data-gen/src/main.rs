#![allow(unused)]

use clap::Parser;
use rand::Rng;
use kdam::tqdm;
use kdam::BarExt;
use std::env::args;
use std::thread;
use std::sync::mpsc;
use csv::WriterBuilder;
use std::error::Error;

/// Program to generate data for the bianry classifier
#[derive(Parser, Debug, Clone)]
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

    /// Number of threads to use
    /// Ensure threads number divides dataset size
    #[arg(short, long, default_value_t = 1)]
    threads: i64,
}

/// generate a random sequence
fn generate_random_sequence(args: &Args) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    return (
        0..args.max_length
    ).map(
        |_| rng.gen_range(0..(args.group_size.pow(2)-1))
    ).collect();
}

/// tells you if a sequence is the identity
fn is_identity(seq: &[i64], args: &Args) -> bool {
    // initialize the permutations
    let mut perm: Vec<i64> = (0..args.group_size).collect();
    
    // do the swaps
    for i in seq.iter() {
        // 0 is the identity
        // so we ignore it if we see it

        let x = *i/args.group_size;
        let y = *i%args.group_size;

        if *i != 0 {
            perm.swap(x as usize, y as usize);
        }
    }

    // check if the permutation is the identity 
    for i in perm.iter() {
        if perm[*i as usize] != *i {
            return false;
        }
    }

    return true
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // throw error if inputs are bad
    assert!(
        args.dataset_size % args.threads == 0, 
        "\nThreads must divide dataset size\n"
    );

    // do the general data
    let mut general_data: Vec<Vec<i64>> = Vec::new();

    // generate the general data
    println!("Generating general data...");
    for _ in tqdm!(
        0..((args.dataset_size as f64)*(1.0-args.identity_proportion)) as i64
    ) {
        general_data.push(generate_random_sequence(&args));
    }

    // generate the identity data
    // create a progress bar
    let identities_needed = ((args.dataset_size as f64)*args.identity_proportion) as i64;
    let mut identity_data: Vec<Vec<i64>> = Vec::new();

    let (sender, receiver) = mpsc::channel();

    println!("Generating identity data...");

    for _ in 0..args.threads {
        let sender_clone = sender.clone();
        let args_clone: Args = args.clone();

        // Spawn a thread
        thread::spawn(move|| {
            let mut count = 0;
            while count < identities_needed / args_clone.threads {
                let seq = generate_random_sequence(&args_clone);

                if is_identity(&seq, &args_clone) {
                    sender_clone.send(seq).unwrap();
                    count += 1;
                }
            }
        });
    }

    // Main thread receives results from worker threads
    for _ in tqdm!(0..identities_needed) {
        // get a sequence and add it to the data
        identity_data.push(receiver.recv().unwrap());
    }

    // Write the data to the file
    println!("Writing data to file...");
    let mut writer = WriterBuilder::new().from_path(args.filename)?;

    for row in tqdm!(general_data.iter().chain(identity_data.iter())) {
        let string_row: Vec<String> = row.into_iter().map(|value| value.to_string()).collect();
        writer.write_record(string_row);
    }

    // Flush the writer to ensure all data is written to the file
    writer.flush()?;

    Ok(())
}