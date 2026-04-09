use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    if args.verbose {
        println!("Cross-Implementation Comparison Tool");
    }
}
