1. Overview
With the rapid emergence of large-scale foundation models featuring hundreds of billions of parameters, efficient inference under constrained computational resources has become a critical challenge for both academia and industry. Achieving high-throughput, low-latency inference at scale requires innovations across system architecture, runtime scheduling, memory management, and kernel-level optimization.

This challenge focuses on ​high-performance inference optimization of large language models under a fixed hardware and software environment​. Participants are required to optimize the inference performance of a given model using the FlagOS inference framework(https://github.com/flagos-ai/vllm-plugin-FL) together with the FlagGems high-performance operator library(https://github.com/FlagOpen/FlagGems).

The goal is to fully exploit hardware capabilities and demonstrate state-of-the-art system-level optimization techniques for large-model inference, achieving significant improvements in throughput and resource utilization while preserving model accuracy.

2. Guidelines
The task centers on optimizing the inference performance of the Qwen3-4B model across multiple hardware platforms using the ​FlagOS inference framework​, with an emphasis on extracting maximal hardware efficiency.

Participants may use their own computing resources and development environments, or they may use the unified computing resources and development environment provided by the organizing committee.
Designing a complete and reproducible optimization solution for large-model inference.Conducting experiments across all official evaluation datasets.Submitting source code and a detailed technical report.
Using FlagOS as the mandatory inference framework to ensure fair comparison, reproducibility, and unified model execution.Submissions that do not comply with the framework requirement will be considered invalid.
2.1 Eligibility and Participation
This track is open to industry teams, universities or research institutes, independent research groups and developers worldwide. Before registration, all participants must complete the Registration Information Form. Registration information must be truthful, accurate, and valid; otherwise, participants may be disqualified from the competition and forfeit any incentives.

Participation rules:

Each team may consist of up to three members (single-person teams are allowed).
Computing Resources and Development Environment: Participants may use their own computing resources and development environment, or utilize the unified computing resources and development environment provided by the organizing committee.
Technical mentorship: Experts from the FlagOS community, including researchers from the Beijing Academy of Artificial Intelligence (BAAI), will provide technical guidance throughout the competition.
Computing Resources Application:

The organizer will provide unified computing resources to selected participating teams. Adhering to the principle of "avoiding waste of computing resources," teams in need of computing resources should submit application materials within the specified period. Application materials must include at minimum: preliminary technical proposals and team introductions (including individual resumes of team members and descriptions of relevant project or competition experience). The organizing committee will allocate computing resources based on comprehensive evaluation results.

Application Channel: Expected to open in early February. Notifications will be sent via email. Teams requiring computing resources must submit application materials by March 11, 2026, to the designated address.
Application Material Requirements: Teams must compile application materials into a zip file not exceeding 20 MB, named with【Computing Application-Team Name】and upload it to the FlagOS platform.
Computing Resources Allocation:

The organizing committee will conduct a comprehensive evaluation of the submitted application materials and make allocation decisions based on the following criteria:

Evaluation Criteria	Weight
Preliminary Technical Proposal:
1. Accuracy of task understanding (20%)
2. Soundness and reproducibility of the technical approach (20%)
3. Clarity of experimental plan and timeline (10%)
4. Degree of innovation or meaningful technical improvements (10%)	60%
Team Background
1. Relevant research or engineering background (15%)
2. Previous competition or project experience (15%)
3. Reasonableness of team role distribution and member stability (10%)	40%
2.2 Timeline
Registration：Jan 9 – May 20
Compute resource application: Jan 9 – Mar 11
Official resource review & allocation: Mar 12 – Mar ​​23
Competition period: Mar 2​4​​ – May 20
Evaluation & review: May 21 – Jun 4
Results announcement & awards: Early June
Any schedule updates will be announced officially. The final timeline is subject to the organizers’ notice.

3. Task Overview
3.1 Competition requirements
This challenge targets extreme inference throughput optimization for the Qwen3-4B model, under the strict constraints of:

Preserving model accuracy (within −2% of the baseline),
Avoiding noticeable latency regression.
Participants are encouraged to apply ​end-to-end system optimization techniques​, including parallelism strategies, memory management, kernel-level computation optimization, and decoding algorithms, to push the performance limits of large-model inference within a fixed framework and hardware environment.

3.2 Evaluation Plan
Evaluation Script: The organizers will provide a unified evaluation script (https://github.com/flagos-ai/vllm-plugin-FL/blob/main/benchmarks/flagos_eval/run_eval.sh). Participating teams should follow the instructions in the script to run and obtain baseline results in their own development environments.
Benchmark Script: The organizers will also provide a unified benchmark script (vllm-plugin-FL/benchmarks/flagos_eval/run_benchmark.sh at main · flagos-ai/vllm-plugin-FL). Participating teams are required to submit the benchmark results, which will be used for ranking.
4. Task Details
4.1 Technical Requirements
Participants must optimize inference for Qwen3-4B based on the FlagOS inference frameworkhttps://github.com/flagos-ai/vllm-plugin-FL

Model references:

HuggingFace: https://huggingface.co/Qwen/Qwen3-4B
ModelScope: https://modelscope.cn/models/Qwen/Qwen3-4B
Modelers: https://modelers.cn/models/Qwen-AI/Qwen3-4B
​Optimization objective​ :

Improve throughput while maintaining model accuracy (≤ 2% degradation),
Ensure latency is not significantly worse than the baseline.
Permitted optimization techniques include (but are not limited to):

Model and tensor parallelism strategies
Memory optimization (e.g., KV cache compression, dynamic batching)
Compute optimization (operator fusion, custom kernel implementation)
Communication scheduling optimization
Model compression
Speculative decoding and advanced sampling algorithms
4.2 Submission Requirements
4.2.1 Required Materials
Submission deadline: March 20, 2026, 23:59 (UTC+8)
Submission materials:
Complete source code
Technical report which should include the detailed methodology, optimizations, and experimental results
4.2.2 Submission Platforms
Submit the source code and technical report on the FlagOS official platform (https://flagos.io/RaceDetail?id=296fmr01&lang=en) and get real-time ranking results
Online benchmarking & leaderboard: FlagOS official website
Final winning submissions (Top 10) will be merged via pull requests into:https://github.com/flagos-ai/vllm-plugin-FL/pulls
4.3 Evaluation Protocol
​Correctness evaluation​:
Official evaluation scripts based on lm-evaluation-harnesshttps://github.com/EleutherAI/lm-evaluation-harness(specific commit and datasets to be announced)
Participants must submit evaluation results to verify correctness and accuracy preservation.
​Performance benchmarking​:
Official vLLM-based benchmarking scripts will be provided.
Metrics include:
Throughput
Latency
4.3.1 Scoring Criteria
​70% ​: Relative throughput improvement over the baseline (e.g., baseline = 200, optimized = 210 → +5% → score = 5% × 70%)
30% : For each additional AI chip successfully adapted with demonstrated performance improvement, participants can receive an additional 30% weighted score.
Evaluation Guidelines:

Scoring Criteria: In cases where multiple submissions achieve identical performance improvements during ranking, the technical innovation of the solution will be assessed based on the technical report submitted by the team (determined by organizer voting).
Model Restrictions: Participants are strictly prohibited from using any models other than Qwen3-4B during the competition. Violation of this requirement will result in disqualification of the submission.
Development Framework Requirement: Participating teams must utilize the FlagOS inference framework as their primary development tool. Final rankings will only be assigned to teams that pass the organizer’s code compliance review. Additional bonus points may be awarded if a team effectively employs and optimizes FlagGems while achieving performance improvement on at least one AI chip provided by the organizer.
Reproducibility and Validity Verification: During the technical solution review phase, the organizer will attempt to reproduce the solutions submitted by participating teams. Solutions that cannot be successfully reproduced will be considered invalid.
5. Awards
​First Prize​: ¥30,000 (1 team)
​Second Prize​: ¥20,000 (3 teams)
​Third Prize​: ¥10,000 (5 teams)
6. Results Announcement
All evaluations and leaderboard rankings will be published on the ​FlagOS official website​. Winners will be notified through official channels.