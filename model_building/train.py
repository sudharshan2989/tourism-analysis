## Download the train and test files from Hugging Face Hub
from huggingface_hub import hf_hub_download

# Define repo and filenames
repo_id = "sudharshanc/tourism-analysis"

Xtrain_path = hf_hub_download(repo_id=repo_id,repo_type="dataset", filename="Xtrain.csv")
Xtest_path  = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="Xtest.csv")
ytrain_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="ytrain.csv")
ytest_path  = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="ytest.csv")


Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = ["Age", "MonthlyIncome", "DurationOfPitch", "NumberOfTrips"]
categorical_nominal = ["Occupation", "Gender", "MaritalStatus", "ProductPitched", "TypeofContact"]
categorical_ordinal = ["CityTier", "PreferredPropertyStar", "Designation", "PitchSatisfactionScore"]
binary_features = ["Passport", "OwnCar"]

# Define category orders
citytier_order = [1, 2, 3]
star_order = [1.0, 2.0, 3.0, 4.0, 5.0]
designation_order = ["Executive", "Senior Manager", "Manager", "AVP", "VP"]
pitchscore_order = [1, 2, 3, 4, 5]

# Scaling Numerical features
numeric_transformer = make_pipeline(
    StandardScaler()
)


categorical_nominal_transformer = make_pipeline(
    OneHotEncoder(handle_unknown="ignore")
)

categorical_ordinal_transformer = make_pipeline(
    OrdinalEncoder(categories=[
        citytier_order,       # applied to CityTier
        star_order,           # applied to PreferredPropertyStar
        designation_order,    # applied to Designation
        pitchscore_order      # applied to PitchSatisfactionScore
    ])

)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat_nom", categorical_nominal_transformer, categorical_nominal),
        ("cat_ord", categorical_ordinal_transformer, categorical_ordinal)
    ]
)

# Final pipeline with classifier
clf = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)


# Define parameter grid for RandomForest
param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [None, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 5],
    "randomforestclassifier__min_samples_leaf": [1, 2]
}

# MLflow tracking
with mlflow.start_run():
    # Grid Search with F1 scoring
    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1"   # <-- use F1 score
    )
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_f1", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Classification metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train, average="weighted")
    test_precision = precision_score(ytest, y_pred_test, average="weighted")

    train_recall = recall_score(ytrain, y_pred_train, average="weighted")
    test_recall = recall_score(ytest, y_pred_test, average="weighted")

    train_f1 = f1_score(ytrain, y_pred_train, average="weighted")
    test_f1 = f1_score(ytest, y_pred_test, average="weighted")

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1
    })

