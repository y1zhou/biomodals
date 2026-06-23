# Preserve PPIFlow candidate sets inside static stage nodes

PPIFlow workflow nodes keep the DAG static but process the full candidate set inside each stage node. LigandMPNN, Partial, ReFold, DockQ preparation, Rosetta analysis, ranking, and reporting must preserve candidate identity across all derived artifacts. Nodes may narrow candidates only through explicit selector configuration, not by silently taking the first structure or relying on sorted file order.
