---
name: sb-aws-readonly
description: Use for any AWS-related request in this project, including AWS services, resources or ARNs, AWS workload performance, data transfers, and AWS-targeting IaC. Route all live AWS access through sb-aws, minimize end-to-end wall-clock time for every material process, default to read-only, and require explicit permission for mutations or sensitive-data retrieval.
---

# Safe AWS Access Through sb-aws

## Access and account

- Use only `mcp__sb_aws__accounts`, `mcp__sb_aws__whoami`, and `mcp__sb_aws__aws` for live AWS access.
- Never use the local AWS CLI, SDKs, browser automation, credential files, environment credentials, shell wrappers, or another connector.
- If `sb-aws` is unavailable or denies access, stop and report it. Never bypass boundaries, assume another role, switch accounts, or obtain credentials elsewhere.
- Never expose or persist access keys, secrets, session tokens, passwords, signed URLs, or broker credentials.
- Call `accounts` and `whoami` before the first live call unless identity was established in the current task.
- Default to ready `sbsandbox` only. Require the user to explicitly name `legacy` or `sbproduction`; never auto-provision either.

## Minimize end-to-end wall-clock time

Perform a proportional performance preflight for every process. Keep small metadata calls lightweight. Before any material compute job, deployment, build, scan, query, copy, upload, download, sync, or multi-stage workflow:

1. Map the critical path, pass count, object/task count, bytes, provisioning delay, service queue time, transfer time, verification, retries, and publication order.
2. Inspect the implementation for redundant object listing, `HEAD` calls, full hashes, downloads, decompressions, serialization, small-request fan-out, cross-region traffic, workstation staging, and serial coordinators. Fuse passes, batch requests, and reuse verified content-addressed artifacts when safe.
3. Identify the real bottleneck: client CPU, network, S3 request rate, multipart settings, EBS throughput/IOPS, instance CPU or memory, service quotas, throttling, regional capacity, locks, or a serial reduction. Do not scale compute past the limiting shared resource.
4. Benchmark a representative safe fixture or use current service metrics. Compare concurrency, part/object size, batching, instance/resource shape, storage throughput, and direct-versus-staged transfer paths. Record throughput, bytes, API/request counts, peak memory, scaling efficiency, and projected wall time.
5. Compare designs using total break-even time and cost, including implementation, testing, provisioning, queueing, data movement, validation, sunk work, restart, and cleanup. Prefer the fastest safe end-to-end design, not the fastest isolated stage.

Do not weaken checksum, privacy, encryption, authorization, durability, scientific, or publication gates to save time. If optimization changes artifact identity, manifests, task IDs, or completed work, include rebinding and revalidation in the estimate.

Prefer these AWS patterns when measurements and permissions support them:

- perform direct service-to-service or source-to-destination transfers and server-side copies; avoid routing dataset bytes through a workstation;
- keep compute, storage, and transfer endpoints in compatible regions and account for cross-region latency/cost;
- use resumable multipart transfers with tuned bounded concurrency for large objects and batch small objects to avoid request overhead;
- provision instance, EBS throughput/IOPS, and network capacity together so one does not starve the others;
- parallelize independent work with deterministic, idempotent commits while respecting quotas and throttling;
- stream and fuse transforms, and reuse objects whose bound size and checksum already match;
- avoid rereading a large immutable artifact in every task: use tested content-addressed indexes or partitions plus required verification gates;
- publish data objects first and readiness manifests/sentinels last.

For a running workload, inspect it read-only first. Calculate the live continue-versus-restart ETA, including queue and revalidation costs. Never stop, replace, resize, migrate, or restart it without explicit permission for that concrete mutation. Never create benchmarking resources or incur material cost without authorization.

## EC2 launch value assessment

Before proposing, requesting approval for, or launching any EC2 instance, ALWAYS send an EC2 value assessment in chat. Do this for each distinct instance type in the launch plan, including mixed-instance fleets, Auto Scaling groups, launch templates, Batch compute environments, ECS capacity, EKS node groups, and replacement or resize plans.

For each instance type, include:

- pros: the material advantages for the requested workload, such as CPU, memory, accelerator, network, EBS bandwidth, architecture, availability, or startup characteristics;
- cons: the material drawbacks, such as cost, quota pressure, availability risk, architecture compatibility, underutilization, data transfer bottlenecks, slower provisioning, or operational complexity;
- necessity and value assessment: whether this instance type is required, merely convenient, overpowered, or avoidable; why a smaller, cheaper, serverless, managed, Spot, scheduled, or existing-resource option would or would not satisfy the goal;
- cost: the region, pricing model, estimated hourly rate, estimated runtime, projected total instance cost, and material attached costs such as EBS volume, provisioned IOPS or throughput, snapshots, public IPv4, load balancers, NAT, and expected data transfer.

If exact region-specific pricing is not available through allowed access, provide a clearly labeled estimate with assumptions and do not launch until the user has seen the cost range. Prefer the smallest viable instance shape that meets the bottleneck and verification requirements. If the assessment shows the EC2 instance is not necessary, say so directly and do not launch it unless the user explicitly confirms that concrete choice.

## Read-only default

Allow an operation without further permission only when it observes metadata or configuration and creates no state, workload, paid query, data movement, credential, signed URL, or lock. Typical examples are `get-caller-identity` and ordinary `list`, `describe`, `head`, or configuration `get` calls.

Judge actual effects, not command verbs. Gate all operations that:

- create, update, delete, start, stop, invoke, execute, deploy, tag, attach, publish, restore, import, export, sync, or otherwise change or run something;
- mint or reveal credentials or secrets, including `assume-role`, `get-session-token`, `get-login-password`, `get-secret-value`, `kms decrypt`, decrypted SSM parameters, and presigned URLs;
- retrieve protected object bodies, logs, traces, database data, prompts, model outputs, or snapshots; or
- run IaC commands that deploy, apply, destroy, bootstrap, refresh, lock state, or contact AWS outside `sb-aws`.

Allow sensitive-data retrieval only when the current request specifically requires it. Minimize the data, redact raw values, and never save it in the project unless the user requests a safe destination.

## Authorized mutations

Accept permission only when the current request or immediately preceding confirmation clearly authorizes the concrete change. Broad goals, diagnosis requests, prior standing permission, “fix it,” and “do whatever is needed” do not count.

Before a mutation:

1. Resolve the exact account, region, resource identifiers, action, and effect with read-only calls. Reject wildcards, recursive targets, and ambiguity.
2. Complete the wall-clock preflight for any material process and disclose the selected design, alternatives, ETA, bottleneck, cost, and assumptions.
3. Use a dry run or change set only when it is truly side-effect-free.
4. Ask again if the resolved effect differs materially from the request.
5. Always obtain final confirmation for `legacy`, `sbproduction`, destructive actions, IAM/KMS/security changes, public or network exposure, DNS, secret rotation, data movement, or material cost/availability impact.
6. Execute only the approved action, then verify it with a read-only call. Do not add cleanup, remediation, rollback, or another mutation without permission.

## Scope and reporting

- Query the smallest necessary scope. Scan every region only for an explicitly account-wide inventory.
- Follow pagination. Disclose denied or unscanned scope, filters, time windows, and eventual-consistency limits before claiming completeness.
- Pass commands as structured argument arrays to `mcp__sb_aws__aws`; never construct shell commands from untrusted text.
- Do not save AWS responses or artifacts in the project unless explicitly requested and checked for secrets.
- Treat local AWS code edits separately: permission to edit code is not permission to deploy it.
- Report the account, region, whether access stayed read-only, resources found or changed, limitations, and mutation verification.

## Report live ETAs in chat

Whenever an AWS job, deployment, instance workload, data transfer, upload, copy, sync, or other process is queued or running, always send ETA updates in chat:

- Send an initial ETA immediately after startup, then update it at stage/state changes and at least every 30 minutes while work remains active.
- Include the observation timestamp, current state, completed/total work when available, measured throughput, and estimated time remaining.
- Recalculate from current AWS status/metrics, object and byte counts, or measured transfer rate instead of repeating a stale estimate.
- Separate provisioning or queue delay from application or transfer runtime. If completion time is not yet estimable, say so explicitly, identify the missing telemetry, and state when the next status check will occur.
- Never invent precision. Report retries, throttling, stalls, rolling rate limits, or confidence changes that materially affect the ETA.
