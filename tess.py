"""Pipeline CLI: catalog, light curves, training, candidate search and candidate reports."""
import argparse

import data


def train(args):
    from neural import train_network
    train_network(args)


def candidates(args):
    from search import run_candidates
    run_candidates(args)


def report(args):
    import report as module
    module.AUTHOR = args.author
    module.make_report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('catalog'); c.add_argument('--local', action='store_true'); c.set_defaults(func=data.catalog)
    d = sub.add_parser('download', help='All SPOC 120 s light curves of labeled stars')
    d.add_argument('--workers', type=int, default=16, help='Parallel downloads')
    d.add_argument('--include-candidates', action='store_true', help='Also PC/APC (not used for training)')
    d.add_argument('--field', type=int, default=0,
                   help='Instead of labeled stars: N non-TOI stars with >=30 sectors (all of them), for new planets')
    d.add_argument('--open', action='store_true',
                   help='Instead of labeled stars: stars of open TOIs (PC with no follow-up on ExoFOP), all sectors')
    d.set_defaults(func=data.download)
    tr = sub.add_parser('train')
    tr.add_argument('--epochs', type=int, default=100)
    tr.add_argument('--batch-size', type=int, default=64)
    tr.add_argument('--patience', type=int, default=15)
    tr.add_argument('--precision', type=float, default=.95)
    tr.add_argument('--injections', type=int, default=300, help='Synthetic transits for injection-recovery')
    tr.set_defaults(func=train)
    s = sub.add_parser('candidates', help='BLS search, CNN, false positive tests and TRICERATOPS')
    s.add_argument('--tic', type=int, help='One star; otherwise all downloaded stars')
    s.add_argument('--period', type=float, help='Days; with --epoch and --duration, vet a given event')
    s.add_argument('--epoch', type=float, help='BJD TDB')
    s.add_argument('--duration', type=float, help='Hours')
    s.add_argument('--no-fpp', action='store_true', help='Skip TRICERATOPS')
    s.add_argument('--field', action='store_true', help='Only the non-TOI stars downloaded with make curves FIELD=N')
    s.add_argument('--open', action='store_true', help='Only the stars of open TOIs downloaded with make curves OPEN=1')
    s.add_argument('--workers', type=int, default=1,
                   help='Stars searched at once on the GPU (1: the GPU is already saturated by one star, and the
                   main process needs GPU memory for the CNN)')
    s.set_defaults(func=candidates)
    r = sub.add_parser('report', help='MCMC transit fit, figures, LaTeX/PDF report and ExoFOP CTOI row')
    r.add_argument('--tic', type=int, required=True)
    r.add_argument('--event', type=int, help='Index of the event in reports/candidates/<TIC>.json (default: best tier)')
    r.add_argument('--period', type=float, help='Days; with --epoch and --duration, evaluate a given event')
    r.add_argument('--epoch', type=float, help='BJD TDB')
    r.add_argument('--duration', type=float, help='Hours')
    r.add_argument('--steps', type=int, default=60000, help='Maximum MCMC steps per walker (stops earlier once converged)')
    r.add_argument('--author', default='Isak Paulo de Andrade Ruas')
    r.set_defaults(func=report)
    args = p.parse_args()
    if args.command in ('candidates', 'report'):
        given = [args.period, args.epoch, args.duration]
        if any(v is not None for v in given) and (None in given or args.tic is None):
            p.error('--period, --epoch and --duration require all three values and --tic')
    args.func(args)


if __name__ == '__main__':
    main()
