CLASSIFICATOR_ALGORITHMS = {
    "algorithms": [{
        "name": "AdaBoostClassifier",
        "parameters": [
            {
                "flag": "estimator",
                "description": "The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.",
                "required": False,
                "types": [
                    {
                        "name": "model"
                    }
                ]
            },
            {
                "flag": "n_estimators",
                "description": "The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. Values must be in the range [1, inf).",
                "required": False,
                "types": [
                    {
                        "name": "int",
                        "from": 1,
                        "closed": "left"
                    }
                ]
            },
            {
                "flag": "learning_rate",
                "description": "The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. Values must be in the range [1, inf).",
                "required": False,
                "types": [
                    {
                        "name": "float",
                        "from": 0,
                        "closed": "neither"
                    }
                ]
            },
            {
                "flag": "random_state",
                "description": "Controls the random seed given at each estimator at each boosting iteration. Thus, it is only used when estimator exposes a random_state. Pass an int for reproducible output across multiple function calls. See Glossary.",
                "required": False,
                "types": [
                    {
                        "name": "int"
                    }
                ]
            },
            {
                "flag": "base_estimator",
                "description": "The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.",
                "required": False,
                "types": [
                    {
                        "name": "model"
                    }
                ]
            },
            {
                "flag": "algorithm",
                "description": "If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.",
                "required": False,
                "types": [
                    {
                        "name": "enum",
                        "enum_values": [
                            "SAMME",
                            "SAMME.R"
                        ]
                    }
                ]
            }
        ]
    }]
}
