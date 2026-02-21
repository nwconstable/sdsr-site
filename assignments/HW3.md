## CA5
> Revise your presentation on paper P using the feedback from reviewers. Prepare a 500 word narrative summarizing the revisions for suggested improvements from reviewers. Place the revised presentation and narrative on the web and ensure that the course web-site has a link to your material. 

The feedback for our slides was fairly positive overall.  There were, however, a few improvements that the feedback suggested.  Noah revised the slides based on their feedback, and then we reviewed and finalized the design of the edited slides together.

Both reviews asked for more examples and/or imagery to help relate the ideas to real-world scenarios. To address this, Noah added wildfire prediction as an example of an objective that can be approached with either homogeneous data or hetereogeneous data.  The wildfire example included geographic maps for drought data and vegetation data, and he included text that indicated using both datasets would be heterogeneous while only using one would be homogeneous.  Noah also added an image of a chart to the *Contributions: Heterogeneity* and an image of various homogeneous and heterogeneous graph examples to a new *Homogeneous vs Heterogeneous* slide.  I suggested we move the graph examples to the heterogeneity slide since the context was still fitting and the presentation flow was improved.

One critique brought up from Group 3 was to relate *possible* applications to real-world applications.  In response, Noah added a slide about Relational Deep Learning (RDL) and how RDL connects to graph learning.  In addition, he provided the [MovieLens](https://movielens.org/) application as an example of a recommender system built by RDL.  The example included a photo of members of the team, and a glance of the website hosting the application.

Both reviews asked for more specific details across the following topics: message passing, the explainer callback mechanics, and temporal graph sampling. In response, Noah added short bullet-points about the EdgeIndex tensor, PyG's temporal graph sampling, and the `Explainer` callback mechanism within the message passing process.


## CA6
> Submit two questions to test the understanding of key concepts in P in a closed-book examination. Attach model solutions to each questions. Each question should be answerable in 10 minutes. Questions should be problem-solving in nature with a unique answer. Questions can be similar to the quiz from the presentation. If similar, provide a 75-100 word justification of the answer, explaining why the choices are correct or incorrect.

### Question 1: Graph modeling and heterogeneity (wildfire prediction)

You are building a GNN to predict large wildfires in the United States. You have the following data sources:
- Drought index per county over time
- Vegetation type per region
- Köppen climate type per region
- Historical wildfire incidents with location and date

You must design a *heterogeneous* graph schema for PyG 2.0 that best captures causal structure for prediction.  Which of the following node/edge schema is **most appropriate** for a scalable heterogeneous GNN in PyG 2.0?

- (A) A single node type, “Region”, with all features as node attributes and no edges.

- (B) Node types of “Region” and “Wildfire” with edges of type `Region–Region` for adjacent regions, and `Region–Wildfire` when wildfires occurred in region.  The measures of drought, vegetation and climate type will be regional node features.

- (C) Node types of “Drought”, “Vegetation”, “Climate” and “Wildfire”.  Edges between nodes such as `Wildfire-Climate` and `Wildfire-Wildfire` represent shared timestamps when present.

- (D) A single node type of “Wildfire” with edges between wildfires in chronological order. All other data will be node features.

*Answer: (B).*

*Reasoning:  Option B cleanly separates entities (regions, wildfires) into node types and uses edges to encode meaningful relations: spatial adjacency (Region–Region) and event occurrence (Region–Wildfire). This matches PyG 2.0’s heterogeneous and temporal capabilities, where “entities become nodes, primary-foreign key links become edges” and “heterogeneous graph data types, message passing… temporal graph handling” are first-class. By placing drought, vegetation, and climate as features on “Region” nodes, the model can learn causal patterns across space and time while remaining scalable and structurally interpretable.*

*Meanwhile, Option (A) has no edges and is thus not a graph system.  Option (D) is a homogeneous graph structure.  Option (C) fails to connect any nodes without a time component, such as climate and vegetation.*

### Question 2: Substructure attribution and explainer choice

A PyG 2.0 explainer is applied to a social-network GNN that predicts whether a user is likely to commit fraud. The explainer outputs the following details:

    The edge between **Tom** and **Robert** is highlighted as highly relevant

    For **Tom**, the feature interest = "bitcoin" is marked highly relevant; location = Istanbul is marked irrelevant

    For Robert, the feature interest = "crime", "fraud" is marked highly relevant

You are told that the explainer works by learning a differentiable mask over edges and node features, and then re-running the model with masked messages to see if the prediction changes as little as possible.

1) Which substructure attribution technique is being used?  

- (A) Feature Importance
- (B) SHAP
- (C) GraphMask
- (D) Counterfactual Reasoning

2) Which aspects of the GNN are directly exploited by this explainer?  Choose the most complete option.

- (A) Only node features
- (B) Only edge structure
- (C) Hidden states and messages in message passing
- (D) Global graph-level pooling only

*Answers: (C) and (C).*

*Reasoning:*

*(1) Key descriptions such as “learning a mask,” “masking messages,” and “re-computing the forward pass without changing the output” are part of GraphMask.  GraphMask uses vertex hidden states and messages at layer 'k' to predict a mask and re-compute the forward pass with modified node states.  Feature Importance (A) and SHAP (B) typically operate on features or inputs, not learned message masks. Counterfactual reasoning (D) would explicitly change outcomes.*

*(2) Options (A), (B) and (D) are disqualified since the mask is learned over both nodes and edges.  The explainer leverages hidden states and messages in the message-passing layers, so (C) is uniquely correct.*

## CP4
> Possible outline or overall structure for paper / project. Do include a justification comparing and contrasting your proposed paper / project with the summary of key readings. There should be a clear statement of the new aspects of your work. Look at the "literature survey and our contribution" sections of the papers in the reading list for examples. 

## CP5
> ### Midterm Slides: 
> The slides should summarized Heilmeir Questions such that each questions should be addressed in 1-2 slides.
> The questions are as follows:
> - What are you trying to do ?: Problem Statement (1 slide)
> - If we succeed, what difference do we think it will make ?: Significance of the problem (1 slide)
> - Why is it hard ?: Challenges (1 slide)
> - How does this get done, at present ?: Related Work and Our Contributions (1-2 slides)
> - What is the new technical idea ? : Propsoed Approach (1-2 slide)
> - Validation of listed contribution (experimental, analytical) (1-2 slides)
> - Conclusions and Future Work (1 slide)
> - Weekly Plan and Task (1 slide)
> ### Validation Slide: 
>Choose a methodology similar to the ones used in the papers in the reading list. For example, experiments, analytial methods, case study, detailed illustrative examples, prototyping and demonstration of new capability, etc. There should be a clear plan to list the steps within each methodology. For example an experimental methodology should include a description of the experiment design listing the candidates to be compared, metrics of performance, values of fixed parameters, valuesets for variable parameters, benchmark datasets and computations, key assumptions, etc. ([Proposal example](https://www.spatial.cs.umn.edu/Courses/Spring26/8715/course_project/CP5_proposal.pdf), [Mid-term presentation example](https://www.spatial.cs.umn.edu/Courses/Spring26/8715/course_project/CP5_slides.pdf))
> ###  Weekly Plan Slide: 
> This slide should contain a table with 5 rows and 2 columns, where the column names are 'Weeks' and 'Task'. Each row will be filled with a 'Week' (e.g., 2/25 - 3/3) and a 'Task' (e.g., running experiments, etc.).
> ### Formal Proposal: 
> The proposal should encompass all slides, detailing the Heilmeier questions along with a weekly plan. It should include an introduction that briefly covers all six elements, along with the contributions. 


## DA3
> Peer-review two other drafts, using the [Digital Accessibility Resource Guide](https://docs.google.com/document/d/186tC1DNCltOGQLC5CraoU9jnPf_Z0_Lc5jNHTJ53lEc/edit?tab=t.0#heading=h.yp9oi26yb5op) and [Accessibility Checker](https://support.microsoft.com/en-us/office/improve-accessibility-with-the-accessibility-checker-a16f6de0-2f39-4a2b-8bd8-5ad801426c7f). Write a peer-review summary and submit it. (Team G1 will peer-review teams G2 and G3 / G2 will peer-review G3 and G4 / ... / G8 will peer-review G9 and G1 / G9 will peer-review G1 and G2).

Given the instructions, we will review the summaries for G3 and G4.