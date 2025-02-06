// main.rs
//
// This file is a full translation of the provided Python training script into Rust.
// It uses the tch crate for neural network and tensor operations and the clap crate for argument parsing.
// Note that many distributed features are only represented as placeholders, because full
// distributed training (e.g. using NCCL) is not yet available in tch-rs.
// A dummy pyramidnet model is implemented in the `pyramidnet` module.
// In a production system you would replace this with a full implementation.

use clap::{App, Arg};
use std::env;
use std::thread;
use std::time::{Duration, Instant};

use tch::nn::{self, ModuleT, OptimizerConfig};
use tch::{Device, Kind, Tensor};

/// Dummy pyramidnet implementation.
/// In a full implementation, you would replace this module with the complete pyramidnet architecture.
mod pyramidnet {
    use tch::nn;
    use tch::{Device, Tensor};

    /// The PyramidNet structure containing a sequential module.
    pub struct PyramidNet {
        pub seq: nn::Sequential,
    }

    impl PyramidNet {
        /// Constructs a new PyramidNet.
        ///
        /// # Arguments
        /// * `vs` - A reference to a variable store path.
        pub fn new(vs: &nn::Path) -> PyramidNet {
            // This dummy implementation builds a small convolutional network.
            // In a real pyramidnet, you would build a much deeper and more complex architecture.
            let mut seq = nn::seq();
            // First convolution: from 3 input channels to 16 channels with kernel size 3.
            seq = seq.add(nn::conv2d(
                vs,
                3,
                16,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ));
            seq = seq.add_fn(|xs| xs.relu());
            // Second convolution: from 16 to 32 channels.
            seq = seq.add(nn::conv2d(
                vs,
                16,
                32,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ));
            seq = seq.add_fn(|xs| xs.relu());
            // Flatten the tensor for a linear layer (omitted here for brevity).
            seq = seq.add_fn(|xs| xs.view([-1, 32 * 32 * 32]));
            // A final linear layer could be added here to map to the 10 classes.
            PyramidNet { seq }
        }
    }

    impl nn::ModuleT for PyramidNet {
        /// Forward pass through the network.
        fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.seq.forward_t(xs, train)
        }
    }
}

/// Main entry point of the program.
fn main() {
    // Define and parse command-line arguments using clap.
    let matches = App::new("CIFAR10 Classification with PyramidNet")
        .arg(
            Arg::with_name("lr")
                .long("lr")
                .default_value("0.1")
                .help("Learning rate"),
        )
        .arg(
            Arg::with_name("resume")
                .long("resume")
                .default_value("")
                .help("Path to resume checkpoint"),
        )
        .arg(
            Arg::with_name("batch_size")
                .long("batch_size")
                .default_value("768")
                .help("Batch size"),
        )
        .arg(
            Arg::with_name("num_workers")
                .long("num_workers")
                .default_value("4")
                .help("Number of data loader workers"),
        )
        .arg(
            Arg::with_name("gpu_devices")
                .long("gpu_devices")
                .takes_value(true)
                .multiple(true)
                .help("List of GPU device IDs to use"),
        )
        .arg(
            Arg::with_name("gpu")
                .long("gpu")
                .default_value("")
                .help("GPU id to use"),
        )
        .arg(
            Arg::with_name("dist_url")
                .long("dist-url")
                .default_value("tcp://127.0.0.1:3456")
                .help("URL used to set up distributed training"),
        )
        .arg(
            Arg::with_name("dist_backend")
                .long("dist-backend")
                .default_value("nccl")
                .help("Distributed backend"),
        )
        .arg(
            Arg::with_name("rank")
                .long("rank")
                .default_value("0")
                .help("Rank for distributed training"),
        )
        .arg(
            Arg::with_name("world_size")
                .long("world_size")
                .default_value("1")
                .help("World size for distributed training"),
        )
        .arg(
            Arg::with_name("distributed")
                .long("distributed")
                .takes_value(false)
                .help("Enable distributed training"),
        )
        .get_matches();

    // If the user specified GPU devices, set the environment variable accordingly.
    if let Some(gpu_devices) = matches.values_of("gpu_devices") {
        let devices: Vec<String> = gpu_devices.map(|s| s.to_string()).collect();
        env::set_var("CUDA_VISIBLE_DEVICES", devices.join(","));
    }

    // Determine the number of GPUs to use.
    // Here we check the CUDA_VISIBLE_DEVICES environment variable.
    let num_gpus: usize = if let Ok(devs) = env::var("CUDA_VISIBLE_DEVICES") {
        devs.split(',').filter(|s| !s.is_empty()).count()
    } else {
        1
    };

    // Calculate the effective world size.
    let base_world_size: usize = matches
        .value_of("world_size")
        .unwrap()
        .parse()
        .expect("world_size must be an integer");
    let world_size = num_gpus * base_world_size;

    println!(
        "Launching training on {} GPU(s) with a world size of {}",
        num_gpus, world_size
    );

    // Spawn one thread per GPU device.
    let mut handles = vec![];
    for gpu in 0..num_gpus {
        // Clone the matches so each thread can own its own copy.
        let matches = matches.clone();
        let handle = thread::spawn(move || {
            main_worker(gpu as i32, num_gpus as i32, &matches, world_size as i32);
        });
        handles.push(handle);
    }

    // Wait for all threads to finish.
    for handle in handles {
        handle.join().unwrap();
    }
}

/// The worker function that is spawned for each GPU.
/// This function mimics the main_worker in the Python code.
///
/// # Arguments
/// * `gpu` - The GPU id that this worker should use.
/// * `ngpus_per_node` - The number of GPUs on this node.
/// * `matches` - The parsed command-line arguments.
/// * `world_size` - The total number of processes in distributed training.
fn main_worker(gpu: i32, ngpus_per_node: i32, matches: &clap::ArgMatches, world_size: i32) {
    // Set the device for this thread.
    let device = Device::Cuda(gpu as usize);
    println!("Using GPU: {} for training", gpu);

    // Compute the rank of this process in distributed training.
    let base_rank: i32 = matches
        .value_of("rank")
        .unwrap()
        .parse()
        .expect("rank must be an integer");
    let rank = base_rank * ngpus_per_node + gpu;
    println!("Process rank: {}", rank);

    // Placeholder for initializing a distributed process group.
    // In a complete implementation, you would use the provided dist_url,
    // dist_backend, world_size, and rank to initialize distributed training.
    println!(
        "Initializing distributed process group with backend '{}' at '{}' with world_size {}",
        matches.value_of("dist_backend").unwrap(),
        matches.value_of("dist_url").unwrap(),
        world_size
    );
    // [Distributed initialization would occur here.]

    // Create a variable store for model parameters on the designated device.
    let vs = nn::VarStore::new(device);
    println!("==> Creating model...");
    // Build the pyramidnet model.
    let net = pyramidnet::PyramidNet::new(&vs.root());

    // In the Python code, the model is wrapped by DistributedDataParallel.
    // In this Rust example, we leave the model as is. A proper implementation would
    // include gradient synchronization between processes.

    // Adjust batch size and number of workers per GPU.
    let global_batch_size: i64 = matches
        .value_of("batch_size")
        .unwrap()
        .parse()
        .expect("batch_size must be an integer");
    let batch_size = global_batch_size / (ngpus_per_node as i64);

    let global_num_workers: i64 = matches
        .value_of("num_workers")
        .unwrap()
        .parse()
        .expect("num_workers must be an integer");
    let _num_workers = global_num_workers / (ngpus_per_node as i64);
    // (Note: In this example, we do not spawn multiple OS threads for data loading.)

    // Count the total number of trainable parameters.
    let num_params: i64 = vs.trainable_variables().iter().map(|var| var.numel()).sum();
    println!("The number of parameters of the model is {}", num_params);

    println!("==> Preparing data...");
    // Load CIFAR10 training data.
    // This example uses the tch::vision::cifar module which expects the data to be
    // downloaded in the provided directory ("data" here). The function returns a struct
    // containing train_images and train_labels tensors.
    let dataset = tch::vision::cifar::load("data").expect(
        "Failed to load CIFAR10 dataset. Make sure the data exists in the 'data' directory.",
    );

    // Simulate a distributed sampler: split the dataset based on the rank.
    // The total number of training samples.
    let total_samples = dataset.train_images.size()[0];
    let samples_per_process = total_samples / (world_size as i64);
    let start_idx = rank as i64 * samples_per_process;
    // Make sure the last process gets all remaining samples.
    let end_idx = if rank == world_size - 1 {
        total_samples
    } else {
        start_idx + samples_per_process
    };

    // Narrow (slice) the dataset tensors for this process.
    let train_images = dataset
        .train_images
        .narrow(0, start_idx, end_idx - start_idx);
    let train_labels = dataset
        .train_labels
        .narrow(0, start_idx, end_idx - start_idx);

    // Define the loss function (cross entropy) and the optimizer (SGD).
    // Note: tch-rs provides a cross_entropy_for_logits method directly on tensors.
    let lr: f64 = matches
        .value_of("lr")
        .unwrap()
        .parse()
        .expect("lr must be a float");
    let mut optimizer = nn::Sgd::default()
        .build(&vs, lr)
        .expect("Failed to create optimizer");

    // Start the training loop.
    train(
        &net,
        &vs,
        &mut optimizer,
        &train_images,
        &train_labels,
        batch_size,
        device,
    );
}

/// The training function that runs one epoch over the training data.
///
/// # Arguments
/// * `net` - The neural network model.
/// * `vs` - The variable store containing model parameters.
/// * `optimizer` - The optimizer used to update the model parameters.
/// * `train_images` - The training images tensor.
/// * `train_labels` - The corresponding labels tensor.
/// * `batch_size` - The batch size to use.
/// * `device` - The device (GPU) where computations are performed.
fn train<M: nn::ModuleT>(
    net: &M,
    _vs: &nn::VarStore,
    optimizer: &mut dyn nn::Optimizer,
    train_images: &Tensor,
    train_labels: &Tensor,
    batch_size: i64,
    device: Device,
) {
    println!("Starting training...");
    // Set the model to training mode.
    // (Some layers like dropout and batchnorm behave differently during training.)
    let _ = net; // the model reference is used below

    // Total number of samples available.
    let total_samples = train_images.size()[0];
    // Compute the number of batches (using ceiling division).
    let num_batches = ((total_samples as f64) / (batch_size as f64)).ceil() as i64;

    // Variables to track training loss and accuracy.
    let mut train_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;

    // Record the start time of the epoch.
    let epoch_start = Instant::now();

    // Iterate over each batch.
    for batch_idx in 0..num_batches {
        let batch_start = Instant::now();

        // Calculate start and end indices for this batch.
        let start = batch_idx * batch_size;
        let end = std::cmp::min(start + batch_size, total_samples);

        // Get the batch and move it to the specified device.
        let inputs = train_images.narrow(0, start, end - start).to_device(device);
        let targets = train_labels.narrow(0, start, end - start).to_device(device);

        // Forward pass: compute outputs and the loss.
        let outputs = net.forward_t(&inputs, true);
        // The loss is computed using cross entropy directly on the logits.
        let loss = outputs.cross_entropy_for_logits(&targets);

        // Zero gradients, perform backpropagation, and update parameters.
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Update training loss.
        let loss_value = f64::from(&loss);
        train_loss += loss_value;

        // Calculate predictions and update accuracy.
        let predictions = outputs.argmax(1, false);
        // Compare predictions with targets; note that eq_tensor returns a Byte tensor (0/1).
        let batch_correct = predictions
            .eq1(&targets)
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .int64_value(&[]);
        correct += batch_correct;
        total += end - start;

        let acc = 100.0 * (correct as f64) / (total as f64);
        let batch_time = batch_start.elapsed().as_secs_f64();

        // Every 20 batches, print the progress.
        if batch_idx % 20 == 0 {
            println!(
                "Epoch: [{}/{}] | loss: {:.3} | acc: {:.3}% | batch time: {:.3}s",
                batch_idx,
                num_batches,
                train_loss / ((batch_idx + 1) as f64),
                acc,
                batch_time
            );
        }
    }

    let elapsed = epoch_start.elapsed();
    println!("Training time: {:?}", elapsed);
}
