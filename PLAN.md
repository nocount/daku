# daku — Implementation Plan

A one-node Kubernetes cluster on the desktop that runs your own ML work: dwarfgpt training runs, Klazan evals, local model serving and dwarf-bench. It replaces `setsid nohup`, hand-run eval scripts and logs scattered in `~/.cache` with a queue of GPU jobs that keeps its own records.

The name is Klazan for *drum*. Klazan has no word for orchestra, and the drum keeps everyone in time, which is the job here. It replaced the working name *bifrost*, which is already taken by a well-known AI project.

Learning Kubernetes is the side effect, not the product. Each phase fixes one real annoyance, and each fix needs one Kubernetes concept.

Status: **plan, not yet built** (2026-10-28). Target machine: the Ubuntu 26.04 desktop (Ryzen 7800X3D, 29 GB RAM, RTX 5070 Ti 16 GB, driver 595.91 / CUDA 13.2, 1.5 TB free on NVMe).

---

## 1. Goals and non-goals

### Goals
- **Training as a queue.** Submitting a run is one command. A second run waits until the GPU is free and then starts on its own.
- **Runs that survive restarts.** A crash or reboot resumes from the last checkpoint, not from step 0.
- **Eval follows training automatically.** Every finished base run gets a `klazan_eval.py` report in `logs/`, in the same format as today.
- **Reproducibility for free.** The image tag *is* the nanochat commit, so the `.commit` file stops being a manual step.
- **Serving on demand.** GLM-5.3-Flash (or a nanochat checkpoint) sits behind an HTTP endpoint and gives up the GPU whenever training needs it.
- **dwarf-bench on a schedule.** Local models get benchmarked unattended, with API keys kept in Secrets.
- **Learn the basics by using them:**
  - pods, Jobs, Deployments, Services, CronJobs
  - ConfigMaps, Secrets, PV/PVC
  - requests and limits, PriorityClass and preemption
  - Ingress, probes, RuntimeClass and device plugins
  - finally, a custom resource with its own controller

### Non-goals
- Multi-node or HA. One node is the point: there's nothing to hide behind.
- Cloud anything. No EKS or GKE, no managed registry.
- Moving heimdall in. It's a localhost daemon tied to Firefox and systemd, and Kubernetes would add moving parts without teaching anything.
- GPU sharing between jobs. 16 GB of VRAM fits one workload at a time (see §2).
- Replacing `uv run` for interactive work. You still develop and smoke-test on bare metal; the cluster is for long, unattended runs.

---

## 2. Key decisions (and why)

| Decision | Choice | Reasoning |
|---|---|---|
| Distribution | **k3s** | A single binary with containerd, CoreDNS, Traefik ingress, local-path storage and metrics-server built in. It's a real conformant cluster, not a toy like kind-in-docker, and `k3s-uninstall.sh` removes it cleanly. You also don't have docker installed, and k3s doesn't need it. |
| GPU access | **NVIDIA Container Toolkit + k8s-device-plugin**, not the GPU Operator | The driver is already installed and working on the host. The GPU Operator exists to manage drivers on fleets, so here it would only add a dozen pods. The device plugin is the small piece that teaches how `nvidia.com/gpu` becomes a schedulable resource. |
| GPU sharing | **Exclusive: `nvidia.com/gpu: 1` per pod, no time-slicing** | Time-slicing shares compute but not VRAM. GLM pins ~11 GB and d12 training needs most of the card, so sharing would mean OOMs. Exclusivity also gives a free queue: extra pods sit `Pending`. |
| Who gets the GPU | **PriorityClasses: `train` > `eval` > `serve`** | Submitting a training Job preempts the GLM server, which goes back to `Pending` and comes back on its own when the GPU frees up. This is the lesson that makes scheduling click. |
| Image builds | **Podman on the host, imported with `k3s ctr images import`** | This avoids running a registry before you know why you'd want one. An in-cluster registry is a "Later" item. |
| Image tags | **Never `:latest`; tag = git short SHA** | `:latest` makes `imagePullPolicy` default to `Always`, which fails with no registry. SHA tags make every run traceable to code. |
| Storage | **Static `local` PersistentVolumes over existing dirs, bound by PVCs** | The 79 GB in `~/.cache/nanochat` and 93 GB of GLM weights already exist, so nothing gets copied. Manifests still use PVCs like a real cluster would, which teaches PV/PVC binding. |
| Manifests | **Kustomize** (`kubectl apply -k`) for your own YAML, **Helm** only for third-party charts | Raw YAML first, so you see what every field does. Helm templating can hide that while you're learning. |
| Train→eval chaining | **Phase 4: initContainer trick. Phase 7: a controller** | Start with the simplest thing that teaches the pod lifecycle, then replace it with the real pattern. Argo Workflows is skipped: it's a whole platform for a two-step pipeline. |
| Controller language | **Python + kopf** | It's your stack and gives the fastest path to a working operator. kubebuilder (Go) is the canonical route and a good follow-up once the concepts are solid. |
| Sleep during runs | **Host-side `daku-inhibit` service** | Today `systemd-inhibit` wraps each run. Pods can't reach the host's logind, so a small host unit holds an inhibitor lock while any `gpu=true` pod is Running. |
| Namespace | **`daku`** for everything of yours | Keeps `kubectl get all -n daku` readable next to `kube-system`. |

---

## 3. Architecture

```
                         ┌──────────────────────── k3s node (the desktop) ─────────────────────────┐
  you ── kubectl / k9s ──┤  kube-system: coredns, traefik, local-path, metrics-server,             │
                         │               nvidia-device-plugin (advertises nvidia.com/gpu: 1)       │
                         │                                                                         │
                         │  daku:                                                                  │
                         │   Job train-d20-mix10p   [prio train]  ─┐                               │
                         │   Job eval-d20-mix10p    [prio eval]    ├─► GPU (one at a time)         │
                         │   Deployment glm         [prio serve]  ─┘                               │
                         │     └─ Service glm:8080 ── Ingress glm.localhost ── browser / curl      │
                         │   CronJob dwarf-bench ──► http://glm.daku.svc:8080/v1                   │
                         │                       └─► Anthropic API (judge, key from Secret)        │
                         │   (Phase 7) Deployment ablation-controller ── watches Ablation CRs      │
                         │                                                                         │
                         │  PVs (local, on the host's disk):                                       │
                         │   nanochat-home  → /home/wil/.cache/nanochat          (rw)              │
                         │   klazan         → /home/wil/projects/klazan          (ro)              │
                         │   models         → /home/wil/models                   (ro)              │
                         │   bench-results  → /home/wil/projects/dwarf-bench/results (rw)          │
                         └─────────────────────────────────────────────────────────────────────────┘
  host: daku-inhibit.service ── polls for Running gpu=true pods ── holds systemd-inhibit lock
```

---

## 4. Components in detail

### 4.1 Images (`images/`)
- **`nanochat`**: built from `~/projects/nanochat` at the `dwarfgpt` branch head.
  - A uv base image, then `uv sync --frozen --extra gpu` at build time.
  - The torch wheel brings its own CUDA runtime; the NVIDIA runtime injects the driver libraries at start-up, so the base needs no CUDA.
  - Tagged `nanochat:<short-sha>`, with an OCI label holding the full SHA.
  - Runs as uid/gid 1000, so files written to the cache are owned by `wil`, not root.
- **`llama-glm5`**: a multi-stage build.
  - The build stage uses an `nvidia/cuda` *devel* image to compile `unslothai/llama.cpp` branch `glm5next/upstream` with `-DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF`, the same as the host build.
  - The final stage is a CUDA *runtime* image with only the binaries.
  - It teaches why the build and run images differ: tens of GB vs a few hundred MB.
- **`dwarf-bench`**: a uv image with `--extra openai`.
  - No code change should be needed: `AsyncOpenAI()` reads `OPENAI_BASE_URL`, so pointing it at the in-cluster Service makes `openai:<model>` hit llama-server. Check this in Phase 6.

### 4.2 Training Job (`manifests/train/`)
- `restartPolicy: Never`, `backoffLimit: 2` and `podFailurePolicy`, so an OOM exit fails fast instead of retrying a doomed config.
- The entrypoint `train.sh` is the one piece of glue that matters:
  1. It finds the newest `meta_*.json` in `base_checkpoints/<tag>/`.
  2. If one exists, it passes `--resume-from-step <n>`.
  3. A retried pod, or one rescheduled after a reboot, therefore continues instead of starting over.

  This requires `--save-every` on every run. *Lesson: Jobs retry pods, and only idempotent pods make retries safe.*
- `train.sh` still tees output to `logs/<tag>.log` and writes `logs/<tag>.commit` from the image label, so the existing log layout and eval tooling keep working. `kubectl logs` is a convenience, and it disappears when the pod is deleted.
- Flags come from a ConfigMap per run, generated by Kustomize: depth, `--window-pattern L`, `--device-batch-size 8`, `NANOCHAT_DATA_DIR`, `PYTHONUNBUFFERED=1`.
- Pod settings:
  - `resources.limits: {nvidia.com/gpu: 1}`
  - a memory request sized from a measured run
  - an `emptyDir: {medium: Memory}` at `/dev/shm`. The container default is 64 MB, and PyTorch can need more.
  - `runtimeClassName: nvidia`
  - label `gpu: "true"`

### 4.3 Eval
- Phase 4: the training container moves to `initContainers` and `klazan_eval.py` becomes the main container of the same pod.
  - Init containers run to completion, in order, before the main container starts.
  - The pod's effective GPU request is the max across them, not the sum, so it's still one GPU.
- The corpus path needs care. `dwarfgpt/corpus/klazan-tinystories` is a **symlink** to `klazan/data/corpus/tinystories/packed`, and a symlink whose target is outside the mounted volume dangles inside the container. Mount the klazan repo and pass `--corpus` as the real path inside it.
- Phase 7 replaces this with separate train and eval Jobs that the controller creates.

### 4.4 Serving (`manifests/serve/`)
- **Deployment `glm`**, 1 replica, PriorityClass `serve`, `nvidia.com/gpu: 1`.
  - Same flags as `model-soup/glm53/glm53.sh serve`: `-t 8`, `--no-warmup`, `-ncmoe 44`.
  - Weights mounted read-only from the `models` PV.
- **Probes:**
  - startup and readiness probes on llama-server's `/health`, which returns 503 while loading
  - a liveness probe with a generous threshold, so a slow token isn't treated as a hang
- **Memory limits are the interesting part.**
  - mmap'd file pages count against the container's cgroup.
  - A tight `limits.memory` makes the kernel reclaim page cache *inside the cgroup*, which turns 3.5 tok/s into far less.
  - Measure tok/s with `bench.sh` at no limit, at 24 GiB and at 16 GiB, then pick one. *Lesson: limits aren't just "max RAM", they change I/O behaviour.*
- **Service `glm`** (ClusterIP, port 8080) and an **Ingress** at `glm.localhost` via k3s's Traefik. systemd-resolved already resolves `*.localhost` to loopback.
- Later: a second Deployment serving a nanochat checkpoint, scaled to 0 by default.

### 4.5 dwarf-bench CronJob (`manifests/bench/`)
- Nightly at 01:00 PT, `concurrencyPolicy: Forbid`.
- At ~4 tok/s, 43 questions take about an hour, and two overlapping runs would starve each other.
- Runs `dwarf-bench bench --models openai:glm-5.3-flash --concurrency 1`.
  - The concurrency is set to match llama-server's slots; 5 parallel requests would only queue.
- `OPENAI_BASE_URL=http://glm.daku.svc.cluster.local:8080/v1`, plus a dummy `OPENAI_API_KEY`.
- `ANTHROPIC_API_KEY` comes from a Secret, for the judge (`DEFAULT_JUDGE_MODEL` in `judge.py`). The Secret is created from a gitignored `secrets.env` via Kustomize's `secretGenerator` and is never committed.
- Results are written to the `bench-results` PV. **Publishing to nocount.github.io stays manual**: the cluster never pushes to anything public.
- If the `glm` pod was preempted by training, the bench pod's readiness check (`curl /health` in an initContainer, with a timeout) fails the run cleanly instead of hanging for hours.

### 4.6 `Ablation` custom resource and controller (Phase 7)
```yaml
apiVersion: daku.nocount.dev/v1alpha1
kind: Ablation
metadata: {name: d20-mix10p, namespace: daku}
spec:
  depth: 20
  dataDir: base_data_mix10p
  image: nanochat:f0be82e
  saveEvery: 500
  eval: {sections: "bpb,translation,continuations", klazanCommit: "2012314"}
status:                    # written by the controller
  phase: Training          # Pending → Training → Evaluating → Done | Failed
  step: 1500
  valBpb: 0.812
  enToKz: 0.80
  kzToEnRoundTrip: 0.84
```
- The CRD gets `additionalPrinterColumns`, so `kubectl get ablations` prints phase, val bpb and both translation scores. This is the ablation table from the dwarfgpt notes, live.
- The controller (kopf):
  1. creates the train Job and watches it
  2. creates the eval Job when training succeeds
  3. parses `logs/<tag>-klazan_eval.json` into `status`
  4. sets owner references, so `kubectl delete ablation x` cleans up its Jobs, but never the checkpoints on the PV
- Needs a ServiceAccount, Role and RoleBinding scoped to `daku`. *Lesson: RBAC, and why controllers are just loops that reconcile the desired state against the real one.*

### 4.7 Host side (`host/`)
- `daku-inhibit.service` + `daku-inhibit.sh`.
  - It loops every 60s over `kubectl get pods -n daku -l gpu=true --field-selector=status.phase=Running`.
  - While anything matches, it holds `systemd-inhibit --what=sleep:idle`.
  - It's installed by you with sudo, like heimdall's units.

---

## 5. Phased task list

Each phase ends with a **Gate** (what must work) and a **Lesson** (the concept it was really about). Steps marked **(you, sudo)** are yours to run; Claude can't.

### Phase 0: Prerequisites and a baseline
**Goal:** know what bare metal does, so the cluster can be compared against it.

- [ ] Wait until no long run is in flight. Installing the container toolkit restarts nothing on its own, but don't change the NVIDIA stack under a 2.7-day d20 run.
- [ ] Record baselines:
  - d12 tok/s (~82K, from `logs/d12-mix00.log`)
  - GLM tok/s via `model-soup/glm53/bench.sh`
  - d12 eval wall time
- [ ] Decide how host-side Ollama (on 127.0.0.1:11434) and bare-metal `glm53.sh` coexist with the cluster. **Kubernetes can't see GPU memory used outside it.** Simplest rule: stop them before submitting GPU work.
- [ ] Scaffold the repo (§6), `git init`, first commit on `main`.
- **Gate:** baselines written into `docs/baseline.md`.

### Phase 1: Cluster and GPU
**Goal:** a pod can see the 5070 Ti.

- [ ] **(you, sudo)** Install the NVIDIA Container Toolkit from NVIDIA's apt repo. Check that `nvidia-ctk --version` works.
- [ ] **(you, sudo)** Install k3s. Use the `--write-kubeconfig-mode 644` flag, or copy `/etc/rancher/k3s/k3s.yaml` to `~/.kube/config`.
- [ ] Check that k3s found the NVIDIA runtime in its generated containerd config (`/var/lib/rancher/k3s/agent/etc/containerd/config.toml`), and that a `nvidia` RuntimeClass exists. Create the RuntimeClass if not.
- [ ] Install the k8s-device-plugin with Helm, with `runtimeClassName: nvidia` on its DaemonSet.
- [ ] `kubectl describe node` shows `nvidia.com/gpu: 1` under Capacity and Allocatable.
- [ ] Apply `manifests/smoke/gpu-smoke.yaml`: a pod running `nvidia-smi`. Read its logs.
- [ ] Apply two copies of it at once and watch the second sit `Pending` with `Insufficient nvidia.com/gpu` in `kubectl describe`.
- [ ] Install `k9s` and use it from here on.
- **Gate:** `nvidia-smi` output from inside a pod, and a second pod Pending until the first finishes.
- **Lesson:** nodes, pods, container runtimes, RuntimeClass, extended resources and the scheduler's "fit" check.

### Phase 2: Containerise nanochat
**Goal:** the same d4 smoke run you'd do on bare metal, inside a container.

- [ ] **(you, sudo)** Install podman.
- [ ] `images/nanochat/Containerfile` + `images/build.sh`: build `nanochat:<sha>`, then `podman save | sudo k3s ctr images import -`.
- [ ] PVs and PVCs for `nanochat-home` and `klazan`. Use `storageClassName: ""` and `claimRef` so each PVC binds to exactly its PV.
- [ ] A one-off **pod** (not a Job yet) that runs a d4 smoke train for ~50 steps against `base_data_mix00`, as uid 1000, with `/dev/shm` as tmpfs.
- [ ] Check that checkpoints land in `~/.cache/nanochat/base_checkpoints/d4-k8s-smoke/`, owned by `wil`.
- **Gate:** the smoke run's loss curve matches a bare-metal d4 smoke run to within noise, and tok/s is within 3% of bare metal.
- **Lesson:** images and tags, `imagePullPolicy`, volumes, PV/PVC binding, securityContext, env from ConfigMap.

### Phase 3: Training as a Job
**Goal:** submit a d12 run, walk away, and find it finished and logged.

- [ ] `train.sh` with checkpoint auto-resume (§4.2), and logs/commit files in the existing layout.
- [ ] `manifests/train/` as a Kustomize base, plus one overlay per run that sets the tag, depth and data dir. `kubectl apply -k manifests/train/overlays/d12-mix05p-rerun`.
- [ ] `podFailurePolicy`: exit code 137 (OOMKilled) → `FailJob`, don't retry.
- [ ] `ttlSecondsAfterFinished: 604800`, so finished Jobs clean up after a week while logs stay on disk.
- [ ] **Kill test:** `kubectl delete pod` mid-run. The Job makes a new pod, and it resumes from the last checkpoint.
- [ ] **Reboot test:** reboot mid-run. After boot, k3s restarts and the run resumes.
- [ ] `host/daku-inhibit`. **(you, sudo)** install it. Check that `systemd-inhibit --list` shows the lock while a run is going.
- [ ] Queue test: submit two d12 overlays back to back. The second starts when the first ends.
- **Gate:** a full d12 rerun reproduces the original's val bpb to within run-to-run noise, with no hand-holding.
- **Lesson:** Jobs vs pods, `restartPolicy`, `backoffLimit`, `podFailurePolicy`, idempotency, Kustomize overlays.

### Phase 4: Eval follows training
**Goal:** every run ends with a `-klazan_eval.json`.

- [ ] Move training into `initContainers`, with `klazan_eval.py` as the main container (§4.3). Mount the klazan repo read-only and resolve the corpus path to the real directory.
- [ ] Check that a failed training init container fails the pod and never runs the eval.
- [ ] Check that the eval JSON matches the bare-metal eval of the same checkpoint exactly. Eval is deterministic, so any difference is a bug.
- **Gate:** one `kubectl apply` produces a trained checkpoint *and* its eval report.
- **Lesson:** pod lifecycle, init containers, how resource requests combine across containers.

### Phase 5: Serving GLM
**Goal:** `curl http://glm.localhost/v1/chat/completions` works and gets out of training's way.

- [ ] `images/llama-glm5/` multi-stage Containerfile. Build and import.
- [ ] `models` PV (read-only), Deployment, Service, Ingress (§4.4).
- [ ] Memory-limit sweep with `bench.sh` against the Service. Record tok/s at each limit in `docs/serving.md`.
- [ ] PriorityClasses `daku-train` (1000), `daku-eval` (900) and `daku-serve` (100).
- [ ] **Preemption test:**
  1. With `glm` Running, submit a d12 Job.
  2. `glm` should be evicted, the Job should start, and `glm` should go back to `Pending`.
  3. When the Job finishes, `glm` should come back and pass readiness by itself.
- **Gate:** GLM tok/s via the Ingress within 5% of bare metal at the chosen limit, and the preemption test passes.
- **Lesson:** Deployments vs Jobs, Services and cluster DNS, Ingress, probes, memory limits vs page cache, PriorityClass and preemption.

### Phase 6: dwarf-bench on a schedule
**Goal:** a fresh GLM leaderboard row every morning.

- [ ] `images/dwarf-bench/` Containerfile.
- [ ] Confirm `OPENAI_BASE_URL` routing works with the existing `openai` provider. If it doesn't, add a small `local:` provider prefix to dwarf-bench instead, in its own commit there.
- [ ] Secret from a gitignored `secrets.env` via `secretGenerator`, plus `.gitignore` entries.
- [ ] CronJob (§4.5). First run it by hand with `kubectl create job --from=cronjob/dwarf-bench bench-manual-1`.
- [ ] Results PV, and check that `dwarf-bench report` on the host reads the run.
- **Gate:** three nights of unattended runs, including one where training preempted `glm` and the bench failed fast instead of hanging.
- **Lesson:** CronJobs, Secrets vs ConfigMaps, `concurrencyPolicy`, cross-service dependencies.

### Phase 7: The `Ablation` operator
**Goal:** `kubectl get ablations` is the dwarfgpt results table.

- [ ] CRD with an OpenAPI schema, a status subresource and printer columns (§4.6).
- [ ] kopf controller in `controller/`, built into an image and run as a Deployment in-cluster. Start by running it locally with `kopf run`.
- [ ] ServiceAccount, Role and RoleBinding with only what it needs: Jobs, Ablations, Ablation status and Events.
- [ ] Replace the Phase 4 initContainer pattern with controller-created train → eval Jobs.
- [ ] Write Events on each transition, so `kubectl describe ablation` tells the story.
- [ ] Re-express the d12 5/10/20% ablation as three `Ablation` objects. The printed table should match the numbers in the dwarfgpt notes.
- **Gate:** submit d20-mix10p as an `Ablation` and get a filled-in status row ~2.7 days later without touching anything.
- **Lesson:** CRDs, controllers and reconciliation, owner references and garbage collection, RBAC, status subresources.

### Later
- **In-cluster registry.** A `registry:2` Deployment plus k3s `registries.yaml`, so builds push instead of import. This is the natural next step after Phase 2's import dance.
- **GPU metrics.** DCGM exporter + Prometheus + Grafana via kube-prometheus-stack: utilisation, VRAM, temperature per job.
- **SFT stage.** Add an `sft` step to `Ablation` once `chat_sft.py` has a Klazan task.
- **nanochat serving.** A Deployment for `chat_web` on a chosen checkpoint, so friends can try dwarfgpt on your LAN.
- **GitOps.** Flux or Argo CD watching this repo's `main`, so `git push` *is* `kubectl apply`.
- **kubebuilder rewrite** of the controller in Go, for comparison.
- **A second node.** An old laptop as a CPU-only agent, to learn taints, tolerations and node affinity (GPU jobs must stay on the desktop).

---

## 6. Repo layout

```
daku/
  PLAN.md
  README.md
  docs/
    baseline.md          # Phase 0 numbers
    serving.md           # memory-limit sweep
  images/
    build.sh             # build + tag by SHA + import into k3s
    nanochat/Containerfile
    nanochat/train.sh    # auto-resume entrypoint
    llama-glm5/Containerfile
    dwarf-bench/Containerfile
  manifests/
    base/                # namespace, PriorityClasses, PVs/PVCs
    smoke/gpu-smoke.yaml
    train/base/          # Job template
    train/overlays/<run>/
    serve/               # glm Deployment, Service, Ingress
    bench/               # CronJob, secretGenerator (secrets.env gitignored)
    ablation/            # CRD, RBAC, controller Deployment
  controller/            # kopf operator (pyproject.toml, uv)
  host/
    daku-inhibit.sh
    daku-inhibit.service
  .gitignore             # secrets.env, *.local.yaml
```

---

## 7. Failure modes and known holes

| Risk | Impact | Mitigation |
|---|---|---|
| A host process (Ollama, bare-metal `glm53.sh`, a `uv run` smoke test) holds VRAM | The scheduler thinks the GPU is free; the pod OOMs | Rule: stop host GPU users before submitting. Later, an initContainer that fails the pod if `nvidia-smi` shows foreign processes |
| Retried Job restarts from step 0 | Days of compute lost | `train.sh` auto-resume, `--save-every` required, kill and reboot tests in Phase 3 |
| OOMKilled config retried `backoffLimit` times | Wasted hours | `podFailurePolicy` on exit 137 |
| Laptop-style suspend mid-run | Run stalls | `daku-inhibit` |
| Memory limit throttles GLM via page-cache reclaim | Serving gets slower | Phase 5 sweep; serve with a high or no limit |
| kubelet evicts pods under node memory pressure (GLM's page cache plus training) | Surprise kills | Don't run both at once (priorities enforce it); watch `kubectl get events` during Phase 5 |
| A dangling symlink inside a volume | Eval can't find the corpus | Mount real paths (§4.3) |
| Files on the PV written as root | `wil` can't delete checkpoints | `runAsUser: 1000` everywhere |
| Traefik binds host ports 80/443 | Clashes with anything later on 80/443 | Nothing listens there today. Move Traefik with a HelmChartConfig if needed |
| k3s's iptables rules interact with other networking | Odd host networking | heimdall's 127.0.0.1:7878 is unaffected; `k3s-uninstall.sh` reverses everything |
| A Secret leaks via git | API key exposure | `secretGenerator` from a gitignored file; never `kubectl get secret -o yaml` into a committed file |
| Driver update on the host | The device plugin or toolkit breaks | Re-run the Phase 1 smoke pod after every driver update |

---

## 8. Open items / assumptions to confirm
- **API group** for the CRD: `daku.nocount.dev` assumes you're fine using a domain you don't need to own. Any DNS-shaped string works.
- **Priority order.** The bench pod uses no GPU, so it's never preempted itself, but training preempts `glm` underneath it and the bench run fails. Is that right, or should training wait for an in-progress bench to finish? The second option would need the controller, or a lock, to hold the train Job.
- **GitHub.** The repo is `nocount/daku`. The plan assumes commits go straight to `main`, the same as heimdall, with pushing left to you.
- **Phase 0 timing.** Start after the d20-mix10p run if you launch it on bare metal first, or make d20 the Phase 7 gate as written and wait.

---

## Sources
- k3s installation: https://docs.k3s.io/installation
- k3s NVIDIA runtime support: https://docs.k3s.io/advanced#nvidia-container-runtime-support
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA k8s-device-plugin: https://github.com/NVIDIA/k8s-device-plugin
- Kubernetes Jobs (backoffLimit, podFailurePolicy, TTL): https://kubernetes.io/docs/concepts/workloads/controllers/job/
- Init containers: https://kubernetes.io/docs/concepts/workloads/pods/init-containers/
- Pod priority and preemption: https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/
- Local persistent volumes: https://kubernetes.io/docs/concepts/storage/volumes/#local
- CronJobs: https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/
- Custom resources: https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/
- kopf: https://kopf.readthedocs.io/
- kubebuilder book: https://book.kubebuilder.io/
- k9s: https://k9scli.io/
