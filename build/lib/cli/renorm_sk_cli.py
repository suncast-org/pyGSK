from pyGSK import core

def run(args):
    print("Renormalized SK test CLI invoked.")
    print(f"Received args: {args}")

    result = core.run_renorm_sk_test(
        M=args.M,
        N=args.N,
        ns=args.ns,
        seed=args.seed,
        plot=args.plot,
        save_path=args.save_path,
        assumed_N=args.assumed_N,
        log_bins=args.log_bins,
        log_x=args.log_x,
        log_count=args.log_count
    )

    print("Renormalized SK test completed.")
    return result