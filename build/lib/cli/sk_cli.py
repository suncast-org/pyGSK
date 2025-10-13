 
from pyGSK import core

def run(args):
    print("SK test CLI invoked.")
    print(f"Received args: {args}")

    # Call your core SK test function
    result = core.run_sk_test(
        M=args.M,
        N=args.N,
        d=args.d,
        ns=args.ns,
        alpha=args.alpha,
        seed=args.seed,
        plot=args.plot,
        save_path=args.save_path
    )

    # Optionally print or return result
    print("SK test completed.")
    return result