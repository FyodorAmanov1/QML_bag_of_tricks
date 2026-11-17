# Reporting a security issue

Please use GitHub's private vulnerability reporting when it is available:
https://github.com/FyodorAmanov1/QML_bag_of_tricks/security/advisories/new

Do not place credentials, private datasets or exploitable details in a public
issue. For ordinary reproducibility or installation problems, open an issue
with the command, dependency versions and a small non-sensitive example.

Only load trusted joblib/pickle model artifacts: loading can execute Python.
Examples run on local simulators; review a hardware adapter and its credentials,
permissions, shot budgets and costs before submitting jobs. Benchmark outputs
may contain row identifiers, labels, predictions, source snapshots and local
paths. Review them before uploading results from your own project.
