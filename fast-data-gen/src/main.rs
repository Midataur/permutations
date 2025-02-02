// this code is bad

#![allow(unused)]

use clap::Arg;
use clap::Parser;
use rand::rngs::ThreadRng;
use rand::rng;
use rand::prelude::*;
use kdam::tqdm;
use kdam::BarExt;
use std::env::args;
use std::thread;
use std::sync::mpsc;
use std::thread::Thread;
use csv::WriterBuilder;
use std::error::Error;
use rand::seq::SliceRandom;
use rand_distr::{Geometric, Distribution};

const MIN_WINDOW_SIZE: i64 = 3;

/// Program to generate data
#[derive(Parser, Debug, Clone)]
#[command(about, long_about = None)]
struct Args {
    /// Maximum length of a sequence.
    #[arg(short, long, default_value_t = 10)]
    max_length: i64,

    /// Group order to use in practice.
    #[arg(short, long, default_value_t = 5)]
    group_size: i64,

    /// Dataset size.
    #[arg(short, long, default_value_t = 100_000)]
    dataset_size: i64,

    /// Identity proportion.
    /// Must be a value between 0 and 1.
    #[arg(short, long, default_value_t = 0.0)]
    identity_proportion: f64,

    /// Filename to write to.
    /// Don't include file extensions.
    #[arg(short, long, default_value_t = String::from("data"))]
    filename: String,

    /// Number of threads to use.
    /// Ensure threads number divides dataset size.
    #[arg(short, long, default_value_t = 1)]
    threads: i64,

    /// Type of transposition to use.
    /// Can be "general", "elementary", "hybrid", or "binary" (binary).
    /// Binary means general but written using place value notation.
    /// Hybrid means general, but written as two seperate indices
    /// If binary or hybrid, identity proportion must be 0.
    #[arg(short='T', long, default_value_t = String::from("elementary"))]
    transposition_type: String,

    /// Are we planning on scaling up?
    /// If so, allows us to have a different max_group size to group_size.
    /// If scaling is enabled then identity proportion must be zero.
    #[arg(short='S', long, default_value_t = false)]
    scaling: bool,

    /// Maximum hypothetical group order that can be used later.
    /// Used for scaling gen only.
    #[arg(short='M', long, default_value_t = 0)]
    max_group_size: i64,

    /// Use window?
    /// Determines whether a shift window should be used for the permutations.
    /// Used for scaling generator only.
    /// Can't be used at the same time as relabelling.
    #[arg(short='k', long, default_value_t = false)]
    use_window: bool,

    /// How many windows to use.
    /// Requires use_window to be turned on.
    /// window_count must divide group_size.
    /// Only works with elementary tranpositions.
    #[arg(short='w', long, default_value_t = 1)]
    window_count: i64,

    /// Enable partition windows.
    /// Requires use_window to be turned on.
    /// Incompatible with window_count. Only one of the two can be turned on.
    /// Only works with elementary tranpositions.
    #[arg(short='p', long, default_value_t = false)]
    partition_windows: bool,

    /// Use relabelling?
    /// Determines whether index relabelling should be used for the permutations.
    /// Used for scaling generator only.
    /// Can't be used at the same time as window.
    /// Can't be used with elementary transpositions.
    #[arg(short='R', long, default_value_t = false)]
    use_relabelling: bool,
}

fn get_max_group_size(args: &Args) -> i64 {
    return if (args.scaling) {
        args.max_group_size
    } else {
        args.group_size
    };
}

fn get_digits_needed(args: &Args) -> i64 {
    let largest_possible = if (args.scaling) {
        args.max_group_size
    } else {
        args.group_size
    } - 1;

    return largest_possible.ilog2() as i64;
}

/// generate a random sequence
fn generate_random_sequence(args: &Args) -> Vec<i64> {
    let mut rng = rand::rng();
    let upper_bound = if (
        args.transposition_type == "elementary"
    ) {
        args.group_size
    } else {
        args.group_size.pow(2)
    };

    return (
        0..args.max_length
    ).map(
        |_| rng.random_range(0..upper_bound)
    ).collect();
}

/// Converts a sequence from the group_size order to the max_group_size_order.
/// Required for "general" or "binary" transposition types.
fn convert_order(seq: &[i64], args: &Args) -> Vec<i64> {
    let mut new_seq: Vec<i64> = Vec::new();

    for i in seq.iter() {
        let transformed = *i%args.group_size + args.max_group_size*(*i/args.group_size);

        new_seq.push(
            transformed
        );
    }

    return new_seq;
}

/// Shifts a sequence if required.
/// Used for the window method.
fn shift_sequence(seq: &[i64], sized_shifts: Vec<(i64, i64)>, args: &Args, rng: &mut ThreadRng) -> Vec<i64> {
    let mut new_seq: Vec<i64> = Vec::new();

    let max_group_size = get_max_group_size(args);

    for i in seq.iter() {
        let (used_shift, window_size) = sized_shifts.choose(rng).unwrap();

        let shifted = if (args.transposition_type == "elementary") {
            used_shift + (i % window_size)
        } else {
            i + used_shift*(max_group_size + 1)
        };

        new_seq.push(
            shifted
        );
    }

    return new_seq;
}

/// Relabels a sequence if required.
/// Used for the relabelling method.
fn relabel_sequence(seq: &[i64], args: &Args) -> Vec<i64> {
    let max_group_size = get_max_group_size(args);
    
    // generate a random relabelling
    let mut rng = rand::rng();
    let mut relabelling: Vec<i64> = (0..max_group_size).collect();
    relabelling.shuffle(&mut rng);

    let mut relabelled: Vec<i64> = Vec::new();

    // do the relabelling
    for i in seq.iter() {
        // decompose the transposition
        let x = *i/max_group_size;
        let y = *i%max_group_size;

        // find new indices
        let new_x = relabelling[x as usize];
        let new_y = relabelling[y as usize];

        // recombine
        let new_trans = new_x*max_group_size + new_y;

        // push the new one
        relabelled.push(new_trans);
    }

    return relabelled;
}

/// Assumes that transpositions are in general form.
/// Converts them into two seperate indices.
/// This is used if representing bases binaryly
fn convert_to_seperate_indices(seq: &[i64], args: &Args) -> Vec<i64> {
    let mut new_seq: Vec<i64> = Vec::new();
    
    let mut x: i64;
    let mut y: i64;

    let max_group_size = get_max_group_size(args);

    for i in seq.iter() {
        x = *i/max_group_size;
        y = *i%max_group_size;

        new_seq.push(x);
        new_seq.push(y);
    }

    return new_seq;
}

/// Converts a sequence into binary digits.
/// Resulting sequence will be little endian
fn convert_binary(seq: &[i64], args: &Args) -> Vec<i64> {
    let mut new_seq: Vec<i64> = Vec::new();
    
    for i in seq.iter() {
        // do some funky bitwise operations to extract the binary
        let mut x = *i;
        let digits_needed = get_digits_needed(args);
    
        for j in 0..(digits_needed) {
            // extracts the xth digit of i
            new_seq.push(x & 1);
            x = x >> 1;
        }
    }

    return new_seq;
}

// i have liberated myself from the shackles of good programming
// based on (Fristedt, 1993)
// https://stackoverflow.com/questions/2161406/how-do-i-generate-a-uniform-random-integer-partition
fn rand_partition(args: &Args) -> Vec<i64> {
    let k = f64::exp(-std::f64::consts::PI/f64::sqrt(args.group_size as f64));

    let mut rng = rand::rng();
    let dists: Vec<Geometric> = (3..=args.group_size).map(
        |x| Geometric::new(1.0-k.powf(x as f64)).unwrap()
    ).collect();

    loop {
        let sample: Vec<i64> = dists.iter().map(
            |x| x.sample(&mut rng) as i64
        ).collect();

        let total: i64 = sample.iter().zip(
            3..=args.group_size
        ).map(
            |x| x.0*x.1
        ).sum();

        if total == args.group_size {
            return sample;
        }
    }
}

fn get_permutation(seq: &[i64], args: &Args) -> Vec<i64> {
    let max_group_size = get_max_group_size(args);

    // initialize the permutations
    let mut perm: Vec<i64> = (0..max_group_size).collect();
    
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
            x = *i/max_group_size;
            y = *i%max_group_size;
        }

        if *i != 0 {
            perm.swap(x as usize, y as usize);
        }
    }

    return perm;
}

/// Tells you if a sequence is the identity
fn is_identity(perm: &[i64], args: &Args) -> bool {
    // check if the permutation is the identity 
    for i in perm.iter() {
        if perm[*i as usize] != *i {
            return false;
        }
    }

    return true
}

fn check_inputs(args: &Args) {
    assert!(
        args.dataset_size % args.threads == 0, 
        "\nThreads must divide dataset size\n"
    );

    assert!(
        !(args.scaling && args.group_size > args.max_group_size),
        "\nGroup size can't be lower than max_group_size\n"
    );

    assert!(
        !(args.max_group_size <= 0 && args.scaling),
        "\nmax_group_size must be at least 0\n"
    );

    assert!(
        !(args.scaling && args.identity_proportion > 0.0),
        "\nScaling mode must have identity_proportion = 0\n"
    );

    assert!(
        !(args.use_window && args.use_relabelling),
        "\nYou can't use window and relabelling at the same time\n"
    );

    assert!(
        !(args.use_relabelling && args.transposition_type == "elementary"),
        "\nYou can't use relabelling with elementary transpositions.\n"
    );

    assert!(
        !(args.window_count > 1 && args.group_size % args.window_count != 0),
        "\nWindow count must divide the group size.\n"
    );

    assert!(
        !(args.window_count > 1 && args.transposition_type != "elementary"),
        "\nOnly elementary tranpositions support window_count > 1.\n"
    );

    assert!(
        !(args.window_count > 1 && args.partition_windows),
        "\nCannot have window_count > 1 while partition_windows are enabled.\n"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // throw error if inputs are bad
    check_inputs(&args);

    // do the general data
    let mut general_data: Vec<Vec<i64>> = Vec::new();
    let mut perms: Vec<Vec<i64>> = Vec::new();

    let mut rng = rand::rng();

    // generate the general data
    println!("Generating general data...");
    for _ in tqdm!(
        0..((args.dataset_size as f64)*(1.0-args.identity_proportion)) as i64
    ) {
        // create sequence
        let mut seq = generate_random_sequence(&args);

        // find window shifts if needed
        let mut shifts: Vec<i64> = Vec::new();
        let mut sizes: Vec<i64> = Vec::new();

        if args.use_window {
            let mut partition: Vec<i64> = Vec::new();
            if args.partition_windows {
                partition = rand_partition(&args);
            }

            let window_count = if args.partition_windows {
                partition.iter().sum()
            } else {
                args.window_count
            };

            // get shift sizes
            if args.partition_windows {
                for (pos, amount) in partition.iter().enumerate() {
                    for i in 0..(*amount) {
                        sizes.push(pos as i64 + MIN_WINDOW_SIZE);
                    }
                }
            } else {
                for i in 0..args.window_count {
                    sizes.push(args.group_size/args.window_count);
                }
            }

            // find shifts
            for i in 0..window_count {
                let max_window_shift = get_max_group_size(&args) - sizes[i as usize];

                shifts.push(rng.random_range(0..=max_window_shift));
            }
        }

        let sized_shifts: Vec<(i64, i64)> = shifts.into_iter().zip(sizes).collect();

        // convert for scaling if required
        if (args.scaling) {
            // convert order
            if args.transposition_type != "elementary" {
                seq = convert_order(&seq, &args);
            }

            // shift if needed
            if args.use_window {
                seq = shift_sequence(&seq, sized_shifts, &args, &mut rng);
            }

            // relabel if needed
            if args.use_relabelling {
                seq = relabel_sequence(&seq, &args);
            }
        }

        let mut new_seq = seq.clone();

        // check which version to push
        if (args.transposition_type == "binary" || args.transposition_type == "hybrid") {
            // convert the sequence to individual swaps
            new_seq = convert_to_seperate_indices(&new_seq, &args);

            if (args.transposition_type == "binary") {
                // convert the sequence to binary
                new_seq = convert_binary(&new_seq, &args)
            }
        }
        
        general_data.push(new_seq.clone());

        if (args.scaling) {
            perms.push(get_permutation(&seq, &args))
        }
    }


    let identities_needed = ((args.dataset_size as f64)*args.identity_proportion) as i64;
    let mut identity_data: Vec<Vec<i64>> = Vec::new();
    
    if args.identity_proportion > 0.0 {
        // Generate the identity data
        // Create a progress bar
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

            if (args.scaling) {
                perms.push(perm);
            }
        }
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
    if (args.scaling) {
        let mut perm_writer = WriterBuilder::new().from_path(
            (args.filename.to_string() + "_perms.csv")
        )?;

        for row in tqdm!(perms.iter()) {
            let string_row: Vec<String> = row.into_iter().map(|value| value.to_string()).collect();
            perm_writer.write_record(string_row);
        }
    
        // Flush the writer to ensure all data is written to the file
        perm_writer.flush()?;
    }

    Ok(())
}