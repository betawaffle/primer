# Run explorer

A web page for following training runs as they happen: every eval set's curves with
error bars, the per-held-out-combination and per-generator-group breakdowns, learning
rate, gradient and weight norms, the RoPE gates per layer and head, and several runs
overlaid for comparison. It reads run directories exactly as `primer.train` writes them
(`run.json`, `log.jsonl`, `checkpoints/`) and refreshes every 10 seconds. Runs from before
the richer logging show what they have.

```
python3 tools/viz/server.py --runs /data/primer/runs            # http://127.0.0.1:8080
python3 tools/viz/fake_run.py /tmp/runs/demo --live             # a made-up run that keeps growing
python3 tools/viz/server.py --runs /tmp/runs --port 8091
```

The server uses only the standard library and never writes into a run directory. The
page's state (run, metric, compared runs) is in the URL, so views can be bookmarked.

On the Framework box it runs as a user service in a rootless container
(`primer-viz.service`), on localhost and the Tailscale address, port 8080. From a laptop
on the tailnet: http://framework:8080 once the firewall lets Tailscale traffic in, or
`ssh -L 8080:localhost:8080 framework` and http://localhost:8080 before that.

Next: one sequence at a time from a checkpoint (each token's prediction, pointer
targets and errors), per-token loss to find the confidently wrong tokens, and attention
maps. Those need the model code and torch, which the container image already has.
