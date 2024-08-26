
class Graph_Node_Relation():
    def __init__(self) -> None:
        self.allowed_nodes=["Components",
                            "Class",
                            "English_Name",
                            "System",
                            "Describe",

                            "Variable",
                            "Constant",
                            "Basic_Configuration_Items",
                            "Component_Logic_Item",
                            "Dynamic_Configuration_Items",
                            "Output",

                            "Type",
                            "Default_Value",
                            "Function",
                            "Remark",

                            "Options",
                            "Option_Meaning",
                            "Valid_Interval",
                            "Valid_Interval",
                            "Data",
                            "Table",
                            "Limitation_Factor",
        ],

        self.allowed_relationships=["IS",
                                    "BELONG",
                                    "FOR",
                                    "OF",

                                    "CONSTANT",
                                    "BASIC_CONFIGURATION_ITEMS",
                                    "COMPONENT_LOGIC_ITEM",
                                    "DYNAMIC_CONFIGURATION_ITEMS",
                                    "OUTPUT",

                                    "INCLUDE",
                                    "ENGLISH_NAME",
                                    "DESCRIBE",

                                    "TYPE",
                                    "SUBCLASS_OF",
                                    "EQUIVALENCE",
                                    "DEFAULT_VALUE",
                                    "DESCRIPTION",
                                    "RELATED_TO",
                                    "REMARK",
                                    "LIMITATION_CONDITION",

                                    "FUNCTION",
                                    "OPTIONS",
                                    "WITHOUT"
        ],