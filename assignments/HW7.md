## CP8
> You are already assigned two class projects to review (in teams web page). Review the assigned course projects. You may use the following review form: [Form](https://www.spatial.cs.umn.edu/Courses/Spring26/8715/CSCI-8715_Final_Review.docx). You are expected to submit 500 word comment to authors including a list of 3 to 4 specific suggestions for improvements. For the peer review, BOTH members on a team should provide input to both of the other papers critiques. Do not split the work of reviewing as it reduces feedback to the authors. 

### G3 Review
The problem posed by G3 appears to be that there are misunderstandings about the differences between the terms “positive co-location,” “negative co-location,” “statistically significant co-location,” and “segregation.” The stated significance is that understanding these terms allows us to distinguish between different spatial relationships. The proposed approach is to validate observed patterns against simulated data generated under independence assumptions, and then assess whether the observed participation values are unusually high or low relative to this simulated baseline.

The problem is only partially stated clearly. The output seems to be a decision about whether a co-location pattern is statistically significant, given a chosen significance level α. However, the inputs are not explicitly defined. A reader might infer that the inputs are spatial observations, but the report never specifies what is being observed, in what domain, or under what constraints. This lack of clarity is important, as spatial data can vary widely depending on context and underlying processes. Additionally, key concepts are not properly defined. For example, the “Participation Index” is referenced but never formally introduced or illustrated, making it difficult to fully understand the proposed method.

The importance of the problem is only weakly articulated. While the report mentions general examples such as animal co-habitation and commercial co-location, it does not clearly identify specific real-world applications or stakeholders who would benefit from this work. Nor does it demonstrate where existing methods fail in a way that necessitates the proposed approach.

The report does not explicitly identify key challenges. As a result, there is no clear mapping between challenges and proposed solutions. While an approach is described, it is not framed as addressing specific technical or conceptual difficulties.

The group claims novelty in three main areas: removing user-defined thresholds, incorporating randomness through simulation, and linking spatial relationship terminology to outcomes via their defined “p-values.” However, these claims are not strongly justified. Simulation-based testing and the use of p-values are already well-established in statistical practice, so presenting them as novel contributions is questionable without clearer differentiation from existing work.

Similarly, the report does not convincingly demonstrate superiority over the state of the art. While it claims that introducing p-values improves upon existing approaches, p-values are already central to statistical hypothesis testing. No empirical comparisons, theoretical arguments, or case studies are provided to support claims of improved performance or insight.

The report lists four main contributions: (a) a conceptual distinction between the four spatial relationship terms, (b) the introduction of p-values as a “principled” evaluation method, (c) a random simulation framework under independence assumptions, and (d) formulations for the metrics `p_pos` and `p_neg`. However, the evidence supporting these contributions is limited. The distinctions between the spatial concepts remain unclear and, in some cases, conceptually redundant. The use of “p-values” deviates from standard statistical definitions, leading to potential confusion. Most importantly, the simulation framework is not described in sufficient detail—there is no explanation of how spatial data is generated, what assumptions are enforced, or how density and domain are handled.

The report does include a brief discussion of future work, suggesting improvements to simulation efficiency and extensions to multiple features. However, these ideas are not developed in detail.

Reproducibility is limited. While the formulas for `p_pos` and `p_neg` are clear and could be implemented, the lack of detail regarding the simulation process prevents full replication. Critical elements such as the sampling procedure, spatial domain, and data generation assumptions are omitted.

Finally, the report has several issues with clarity and organization. The Problem Statement introduces confusion about terminology without resolving it, and the Significance section largely repeats earlier points without adding depth. The use of “p-values” does not align with standard definitions, leading to an overloaded and potentially misleading term. Additionally, there are no visualizations or diagrams to support the explanations. While there are no major grammatical issues, the overall presentation lacks cohesion and precision.

### G6 Review (WIP)

> Provide a summary of the report listing the problem (e.g., definition, significance, challenges) and the proposed approach (e.g., description, novelty, superiority over competition).

The group aims to identify co-location patterns among two datasets targetting different areas - retail co-location patterns in Minneapolis and crime co-location in Chicago. The basic idea is to simply find statistically significant relationships that may be used later to identify some unidentified structure informing the co-location, such as commerical demand or urban risk patterns. This in turn can help inform decision-making at the policy level. The approach is easy enough to follow. They load and preprocess the data by picking the 'feature' (brand name, crime category) and the geographic location of that feature to create a neighborhood using a pre-defined threshold value. From this neighborhood, we can find co-location patterns using the participation index. The novelty is not immediately clear to the reader, but seems to be the application of previous works into a unified framework. This framework is then used against different domains and data (commerical in MN, crime in IL) to show the robustness of the process.

> Is the problem stated clearly? 
>- Does it list inputs, output, an objective function, and >constraints (e.g., key assumptions). 
>- Does it illustrate inputs and output.
>- Does it define and illustrate key concepts needed to understand the problem statement?

The problem is stated clearly. The inputs and outputs are not listed inline, but can be inferred as any dataset with similar features and their geographic locations (e.g. commerical brand locations) which leads to a downstream output of the calculated co-location pattern analysis, which is domain-dependent. In section 6, Validation, we see graphics and features of the input datasets used in this paper. The key concepts needed to interpret the paper are suitably defined, whether in the Problem Statement or Validation sections.

>Is the problem's importance articulated by addressing questions such as the following:
>- Who cares about the problem? 
>- If the project is successful, what difference (e.g., societal, technical) will it make?

The problem's importance is articulated well for the given examples. Commerical co-location can help predict consumer demands and patterns, while crime co-location can help inform urban risk mapping. These are domain-specific interpretations, and for each new domain this framework is applied to the interpretation will change. This implies that anyone could find a use for this framework to analyze a problem they care about, and policy-makers could use this to identify problems most people care about. Success of this project at a large scale could help inform policy based on urban risk assessments, commerical patterns, transit demands, residential planning, and more. 

>Are the problem’s key challenges identified? 

There do not seem to be any key challenges identified. The group relies heavily on prior work and the contribution here is to refine that work to an applicable pipeline. This implies the challenge is refining the work and generalizing the principles to be domain-agnostic.

>Is an approach proposed to address the key challenges in the problem?

Sections 5 and 6, Proposed Approach and Validation, show the general process and define it such that any dataset with features attributed to a geographic location could be used with this framework. This shows how the pipeline can be used cross-domain. The results shown in section 6 and discussed in section 7 show clear understanding and effective application of the utilized principles to address the problem.

>Is the novelty of the proposed approach articulated? For example, did the proposal summarize related work and their limitations overcome by the proposed approach?

The novelty is stipulated as the framework itself, and is articulated well enough. The reader would be interested to see an artifact of this work, code to reproduce the results or to apply to other domains. Beyond that, the visualizations and discussions on the results are convincing that the proposed approach accomplished what the group set out to do.

>Did the report articulate the superiority of the proposed approach over the state of the art? For example, did it provide evidence (e.g., examples, theorems, experiments, case-study, etc.).

The main validations listed are case studies using the aforementioned Minneapolis commerical data and Chicago crime data. There are also basic formulaic validations listed in section 6. The investigations of these case studies with the shown framework is effective at convincing the reader of the superiority over state-of-the-art.

>Did the report list contribution claims? For example, did it list contribution such as new concepts, theories, data-structures, algorithms, new approaches, etc.

The group claims the framework for this experiment as a contribution, that is the process of: 
data preprocessing -> spatial feature construction -> neighborhood construction -> candidate generation -> co-location pattern detection
This pipeline seems to be the implied process for the prior works, but never explicitly stated until here. By formalizing the process into a framework with cross-domain applicability, this paper improves on the process itself of state-of-the-art methods.

>Did the report provide evidence to support the contribution claims?

See above

>Did the report include next steps and future work? 

The future work includes three main topics: higher-order co-location pattern analysis, adaptive neighborhood distance thresholds, and temporal analysis. These seem like logical next steps to improve the applicability, robustness, and information that can be gained from this analysis.

>Are the results reproducible? (For example, did the report provide adequate details of proofs, experiment design, case study parameters, assumptions to help readers understand the validation process.)

The project seems to be reproducible. The data is explicitly mentioned, the process is involved enough to follow, and the results are validated and discussed enough to reproduce. The specification of the simulations is a bit weak, but I believe we are told enough in that they are randomized datasets generated within the same bounds as the original (location bounded, number of features) and we are told they are Monte Carlo simulations.

>Other comments (e.g., readability, self-contained?, grammar, adequate use of illustrations, summarizing data with charts, etc.)
- paragraph 5: reword data categories to actual crimes so it fits into the text better: "incidents such as assaults, damage to property, and selling or use of narcotics"

## CP9
> Oral presentation in the class using 10 to 15 slides.
> - Each project presentation should be limited to 25 minutes to allow completion of 3 presentations in a 75 minute meeting. Limit each presentation to less than 12 slides since an average presenter takes about 2 minutes to explain a typical slide (this is a suggestion). Know which slide you should be on at the end of 20 minutes and 22 minutes to ensure proper pace. A partner within each group should watch time and let the speaker know when 5 minutes are left or 2 minutes are left.
> - Presentations on survey papers should include motivation, major problems in the area, key results, open problems, and key sources. Focus on major problems and key results. Use summary figures (e.g. classification diagram for all approached to recovery in the Computing Survey paper in our readings) or tables to highlight key messages.
> - Presentations on projects should follow the format of paper analysis. Candidate sections include motivation, problem definition, key issues and alternative ways of resolving those, related work and their limitations, your approach, validation, conclusions (key contributions), and future work (assumptions and potential extensions).
> - Reviwers may use the following form to review the final class presentations and the final project reports.

[Google Slides](https://docs.google.com/presentation/d/1NOww1ST-1Q10Go-sVE2ZDxI1t4j0GQoASUkMZwjTgPM/edit?usp=sharing)
[PDF](../images/HW7/Project%20Presentation.pdf)
