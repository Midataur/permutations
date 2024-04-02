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
    /// Don't include file extensions
    #[arg(short, long, default_value_t = String::from("data"))]
    filename: String,

    /// Number of threads to use
    /// Ensure threads number divides dataset size
    #[arg(short, long, default_value_t = 1)]
    threads: i64,

    /// Type of transposition to use
    /// Can be "general", "elementary", or "scalable"
    #[arg(short='T', long, default_value_t = String::from("elementary"))]
    transposition_type: String,

    /// Include permutations?
    /// Determines whether to include the actual permutations as well
    /// Used for the scaling generator
    #[arg(short='P', long, default_value_t = false)]
    include_perms: bool,
}

/// generate a random sequence
fn generate_random_sequence(args: &Args) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    let upper_bound = if (
        args.transposition_type == "elementary"
    ) {
        args.group_size
    } else {
        args.group_size.pow(2)-1
    };

    return (
        0..args.max_length
    ).map(
        |_| rng.gen_range(0..upper_bound)
    ).collect();
}

// converts a sequnce to 
fn convert_sequence(seq: &[i64], args: &Args) -> Vec<i64> {
    // assumes that transpositions are in general form
    // converts them into two seperate indices
    // this can be used by the scalable transformer
    let mut new_seq: Vec<i64> = Vec::new();
    
    let mut x: i64;
    let mut y: i64;

    for i in seq.iter() {
        x = *i/args.group_size;
        y = *i%args.group_size;

        new_seq.push(x);
        new_seq.push(y);
    }

    return new_seq;
}

fn get_permutation(seq: &[i64], args: &Args) -> Vec<i64> {
    // initialize the permutations
    let mut perm: Vec<i64> = (0..args.group_size).collect();
    
    // do the swaps
    for i in seq.iter() {
        // 0 is the identity
        // so we ignore it if we see it
        let mut x: i64;
        let mut y: i64;

        if (args.transposition_type == "elementary") {
            x = *i;
            y = *i-1;
        } else {
            x = *i/args.group_size;
            y = *i%args.group_size;
        }

        if *i != 0 {
            perm.swap(x as usize, y as usize);
        }
    }

    return perm;
}

/// tells you if a sequence is the identity
fn is_identity(perm: &[i64], args: &Args) -> bool {
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
    let mut perms: Vec<Vec<i64>> = Vec::new();

    // generate the general data
    println!("Generating general data...");
    for _ in tqdm!(
        0..((args.dataset_size as f64)*(1.0-args.identity_proportion)) as i64
    ) {
        let seq = generate_random_sequence(&args);

        // check which version to push
        if (args.transposition_type == "scalable") {
            // push the converted version
            general_data.push(convert_sequence(&seq, &args))
        }
        else {
            // push the normal version
            general_data.push(seq.clone());
        }

        if (args.include_perms) {
            perms.push(get_permutation(&seq, &args))
        }
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
        thread::spawn(move || {
            let mut count = 0;
            while count < identities_needed / args_clone.threads {
                let seq = generate_random_sequence(&args_clone);
                let perm = get_permutation(&seq, &args_clone);

                if is_identity(&perm, &args_clone) {
                    sender_clone.send((seq, perm)).unwrap();
                    count += 1;
                }
            }
        });
    }

    // Main thread receives results from worker threads
    for _ in tqdm!(0..identities_needed) {
        // get a sequence and add it to the data
        let (seq, perm) = receiver.recv().unwrap();
        identity_data.push(seq);
        perms.push(perm);
    }

    // Write the data to the file
    println!("Writing data to file...");
    let mut seq_writer = WriterBuilder::new().from_path(
        (args.filename.to_string() + ".csv")
    )?;

    for row in tqdm!(general_data.iter().chain(identity_data.iter())) {
        let string_row: Vec<String> = row.into_iter().map(|value| value.to_string()).collect();
        seq_writer.write_record(string_row);
    }

    // Flush the writer to ensure all data is written to the file
    seq_writer.flush()?;

    // write perm data if required
    let mut perm_writer = WriterBuilder::new().from_path(
        (args.filename.to_string() + "_perms.csv")
    )?;

    for row in tqdm!(perms.iter()) {
        let string_row: Vec<String> = row.into_iter().map(|value| value.to_string()).collect();
        perm_writer.write_record(string_row);
    }

    // Flush the writer to ensure all data is written to the file
    perm_writer.flush()?;

    Ok(())
}