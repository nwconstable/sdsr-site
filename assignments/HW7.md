## CP8
> You are already assigned two class projects to review (in teams web page). Review the assigned course projects. You may use the following review form: [Form](https://www.spatial.cs.umn.edu/Courses/Spring26/8715/CSCI-8715_Final_Review.docx). You are expected to submit 500 word comment to authors including a list of 3 to 4 specific suggestions for improvements. For the peer review, BOTH members on a team should provide input to both of the other papers critiques. Do not split the work of reviewing as it reduces feedback to the authors. 

### G3 Review (WIP)
> Provide a summary of the report listing the problem (e.g., definition, significance, challenges) and the proposed approach (e.g., description, novelty, superiority over competition).

The problem posed by G3 appears to be that there are misunderstandings about the differences between the terms “positive co-location”, “negative co-location”, “statistically significant co-location”, and “segregation”. The significance is that understanding these terms allows us to easily distinguish between different spatial relationships. The proposed approach would validate real-world observations against simulations to try to show that the observed participation values are either much higher or lower than you’d see under independent and random scenarios. 

> Is the problem stated clearly? 
>- Does it list inputs, output, an objective function, and >constraints (e.g., key assumptions). 
>- Does it illustrate inputs and output.
>- Does it define and illustrate key concepts needed to understand the problem statement?

The output seems to be a decision of whether a co-location pattern is statistically significant, given some alpha value decided by the tester, but the inputs are not listed. A reader might infer that they are observations of something, but we’re never told what we’re observing, or where. This is important because the things you’re observing can vary drastically from tons of sources, including human intervention. Few concepts are introduced to explain these concepts, such as the mentioned “Participation Index” is never defined. 

>Is the problem's importance articulated by addressing questions such as the following:
>- Who cares about the problem? 
>- If the project is successful, what difference (e.g., societal, technical) will it make?

Some simple cases are proposed, such as animal co-habitation and commercial co-location, but we’re not told of any particularly important cases nor practical uses where currently observed methods lack in analyzing spatial co-habitation. 

>Are the problem’s key challenges identified? 

There is no mention of challenges the group is specifically addressing.

>Is an approach proposed to address the key challenges in the problem?

There are no specified challenges so there are no approaches defined to address them.

>Is the novelty of the proposed approach articulated? For example, did the proposal summarize related work and their limitations overcome by the proposed approach?

The group claims that their approach is novel for the lack of user-defined thresholds (unsupervised objectiveness), accountability for randomness with the random simulator usage, and the strong binding of the terms listed in the Problem Statement to simulated outcomes using their defined “p-values”.

>Did the report articulate the superiority of the proposed approach over the state of the art? For example, did it provide evidence (e.g., examples, theorems, experiments, case-study, etc.).

The group claims to introduce p-values as an improvement over state of the art, but p-values remain the backbone of statistical significance testing, especially against null distributions. 

>Did the report list contribution claims? For example, did it list contribution such as new concepts, theories, data-structures, algorithms, new approaches, etc.

The group lists four main contributions: a) a conceptual distinction between the four previously-listed spatial relationships, b) the introduction of p-values as a “principled” way to evaluate statistical significance, c) a random simulation framework to generate simulated data under independence assumptions, and d) formulations for the metrics `p_pos` and `p_neg` while providing interpretation. 

>Did the report provide evidence to support the contribution claims?

The group does not provide clear definitions nor distinctions between the fours listed co-location patterns. They do introduce p-values, but not as defined in the statistical sense, leading to an overloaded term that confuses the reader. The random simulation framework is not explained at all. We are not told how simulations are created, only that they have an assumption of independence. A simple example using their formulation of `p_pos` and `p_neg` is given, but not expanded on nor explained.

>Did the report include next steps and future work? 

The group includes a short section stating that future work should focus on refinements to the simulation framework in terms of efficiency and robustness. They also claim this can be used for multiple features.

>Are the results reproducible? (For example, did the report provide adequate details of proofs, experiment design, case study parameters, assumptions to help readers understand the validation process.)

The groups formulation of `p_pos` and `p_neg` is reproducible, however the simulation framework is not explained enough to reproduce this, leading to incomplete formulas. We are neither told the sample data nor the sampling methods. The listed co-location patterns are not explained in the paper enough to apply them to any findings.

>Other comments (e.g., readability, self-contained?, grammar, adequate use of illustrations, summarizing data with charts, etc.)

The Problem Statement's ideas are not clearly connected to each other, so the point about confusion between the terms is, itself, confusing. The construction of the paper seems to have redundancies, for example the Significance section seems to be a re-wording of the Problem Statement, with little added especially in regards to the signficance of this problem. P-values are an established part of statistics and the introduction of the term here does not seem to use the common definitions, nor formulations. This leads to an overloaded term that only confuses the reader. There are no visualizations to help convey ideas. There are not any major grammatical issues, only minor typos.

### G6 Review (WIP)

> Provide a summary of the report listing the problem (e.g., definition, significance, challenges) and the proposed approach (e.g., description, novelty, superiority over competition).

> Is the problem stated clearly? 
>- Does it list inputs, output, an objective function, and >constraints (e.g., key assumptions). 
>- Does it illustrate inputs and output.
>- Does it define and illustrate key concepts needed to understand the problem statement?

>Is the problem's importance articulated by addressing questions such as the following:
>- Who cares about the problem? 
>- If the project is successful, what difference (e.g., societal, technical) will it make?

>Are the problem’s key challenges identified? 


>Is an approach proposed to address the key challenges in the problem?


>Is the novelty of the proposed approach articulated? For example, did the proposal summarize related work and their limitations overcome by the proposed approach?


>Did the report articulate the superiority of the proposed approach over the state of the art? For example, did it provide evidence (e.g., examples, theorems, experiments, case-study, etc.).

>Did the report list contribution claims? For example, did it list contribution such as new concepts, theories, data-structures, algorithms, new approaches, etc.

>Did the report provide evidence to support the contribution claims?

>Did the report include next steps and future work? 

>Are the results reproducible? (For example, did the report provide adequate details of proofs, experiment design, case study parameters, assumptions to help readers understand the validation process.)

>Other comments (e.g., readability, self-contained?, grammar, adequate use of illustrations, summarizing data with charts, etc.)

## CP9
> Oral presentation in the class using 10 to 15 slides.
> - Each project presentation should be limited to 25 minutes to allow completion of 3 presentations in a 75 minute meeting. Limit each presentation to less than 12 slides since an average presenter takes about 2 minutes to explain a typical slide (this is a suggestion). Know which slide you should be on at the end of 20 minutes and 22 minutes to ensure proper pace. A partner within each group should watch time and let the speaker know when 5 minutes are left or 2 minutes are left.
> - Presentations on survey papers should include motivation, major problems in the area, key results, open problems, and key sources. Focus on major problems and key results. Use summary figures (e.g. classification diagram for all approached to recovery in the Computing Survey paper in our readings) or tables to highlight key messages.
> - Presentations on projects should follow the format of paper analysis. Candidate sections include motivation, problem definition, key issues and alternative ways of resolving those, related work and their limitations, your approach, validation, conclusions (key contributions), and future work (assumptions and potential extensions).
> - Reviwers may use the following form to review the final class presentations and the final project reports.

[Google Slides](https://docs.google.com/presentation/d/1NOww1ST-1Q10Go-sVE2ZDxI1t4j0GQoASUkMZwjTgPM/edit?usp=sharing)
[PDF](../images/HW7/Project%20Presentation.pdf)
