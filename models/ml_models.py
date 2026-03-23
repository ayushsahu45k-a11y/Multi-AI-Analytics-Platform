import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error,
    r2_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class MLPipeline:
    """
    A powerful, production-ready Machine Learning pipeline supporting
    classification and regression with ensemble methods, cross-validation,
    feature importance, and detailed metrics.
    """

    def __init__(self, task_type: str = "classification", model_name: str = "Random Forest"):
        self.task_type = task_type
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        name = self.model_name
        if self.task_type == "classification":
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, max_depth=None, min_samples_split=2,
                    random_state=42, n_jobs=-1, class_weight='balanced'
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=5,
                    random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42, class_weight='balanced'
                ),
                "SVM": SVC(probability=True, kernel='rbf', random_state=42, class_weight='balanced'),
            }
            return models.get(name, models["Random Forest"])
        else:
            models = {
                "Random Forest": RandomForestRegressor(
                    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
                ),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=1.0, max_iter=5000),
                "SVM": SVR(kernel='rbf'),
            }
            return models.get(name, models["Random Forest"])

    def _preprocess_X(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        df = df.copy()

        # Encode categoricals
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # Boolean → int
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(int)

        arr = df.values.astype(float)

        if fit:
            arr = self.imputer.fit_transform(arr)
            arr = self.scaler.fit_transform(arr)
        else:
            arr = self.imputer.transform(arr)
            arr = self.scaler.transform(arr)

        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        df = df.copy()

        if target_col and target_col in df.columns:
            y_raw = df[target_col]
            if self.task_type == "classification":
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y_raw.astype(str))
                self.classes_ = self.label_encoder.classes_
            else:
                y = y_raw.values.astype(float)
            df = df.drop(columns=[target_col])
        else:
            y = None

        # One-hot for remaining categoricals after splitting target
        df = pd.get_dummies(df, drop_first=True)
        self.feature_names = df.columns.tolist()

        X = self._preprocess_X(df, fit=True)
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the model and return comprehensive metrics."""

        if isinstance(X, pd.DataFrame):
            X = self._preprocess_X(X, fit=True)

        # Stratified split for classification when possible
        stratify = None
        if self.task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) >= 2 and all(c >= 2 for c in counts):
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        self.X_test = X_test
        self.y_test = y_test

        self.model = self._build_model()
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred

        self.metrics = self._compute_metrics(y_test, y_pred, X, y)
        return self.metrics

    def _compute_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        X_full: np.ndarray,
        y_full: np.ndarray,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        if self.task_type == "classification":
            metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average='weighted')), 4)

            # ROC-AUC (binary only)
            if len(np.unique(y_full)) == 2 and hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(self.X_test)[:, 1]
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
                except Exception:
                    pass

            # Cross-validation
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.model, X_full, y_full, cv=cv, scoring='accuracy', n_jobs=-1)
                metrics["cv_mean_accuracy"] = round(float(cv_scores.mean()), 4)
                metrics["cv_std"] = round(float(cv_scores.std()), 4)
            except Exception:
                pass

            # Classification report as string
            try:
                class_names = [str(c) for c in self.classes_] if self.classes_ is not None else None
                metrics["classification_report"] = classification_report(
                    y_test, y_pred, target_names=class_names
                )
            except Exception:
                pass

            # Confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                metrics["confusion_matrix"] = cm.tolist()
            except Exception:
                pass

        else:  # regression
            metrics["mse"] = round(float(mean_squared_error(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["r2_score"] = round(float(r2_score(y_test, y_pred)), 4)

            # Cross-validation
            try:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.model, X_full, y_full, cv=cv, scoring='r2', n_jobs=-1)
                metrics["cv_mean_r2"] = round(float(cv_scores.mean()), 4)
                metrics["cv_std"] = round(float(cv_scores.std()), 4)
            except Exception:
                pass

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        if isinstance(X, pd.DataFrame):
            X = self._preprocess_X(X, fit=False)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_name} does not support probability estimates")
        if isinstance(X, pd.DataFrame):
            X = self._preprocess_X(X, fit=False)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            # Fallback: permutation-style zeros
            importance = np.zeros(len(self.feature_names))

        return pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_predictions_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        """Returns original df with predictions appended."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        result = df_original.copy()
        # Preprocess same features used in training
        feature_df = df_original[[f for f in self.feature_names if f in df_original.columns]]
        preds = self.predict(feature_df)
        result["prediction"] = preds
        return result


# ---------------------------------------------------------------------------
# XGBoost Pipeline
# ---------------------------------------------------------------------------

class XGBoostPipeline(MLPipeline):
    """XGBoost-based pipeline with early stopping and full metrics."""

    def __init__(self, task_type: str = "classification"):
        super().__init__(task_type=task_type, model_name="XGBoost")

    def _build_xgb_model(self, n_classes: int = 2):
        if self.task_type == "classification":
            objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                objective=objective,
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        if not XGB_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        if isinstance(X, pd.DataFrame):
            X = self._preprocess_X(X, fit=True)

        stratify = None
        if self.task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) >= 2 and all(c >= 2 for c in counts):
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        self.X_test = X_test
        self.y_test = y_test

        n_classes = len(np.unique(y)) if self.task_type == "classification" else 2
        self.model = self._build_xgb_model(n_classes=n_classes)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self.metrics = self._compute_metrics(y_test, y_pred, X, y)
        return self.metrics


# ---------------------------------------------------------------------------
# LightGBM Pipeline
# ---------------------------------------------------------------------------

class LightGBMPipeline(MLPipeline):
    """LightGBM pipeline — fastest gradient boosting for large datasets."""

    def __init__(self, task_type: str = "classification"):
        super().__init__(task_type=task_type, model_name="LightGBM")

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        if isinstance(X, pd.DataFrame):
            X = self._preprocess_X(X, fit=True)

        stratify = None
        if self.task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            if all(c >= 2 for c in counts):
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        self.X_test = X_test
        self.y_test = y_test

        if self.task_type == "classification":
            n_classes = len(np.unique(y))
            objective = "multiclass" if n_classes > 2 else "binary"
            self.model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05,
                num_leaves=31, random_state=42,
                objective=objective, n_jobs=-1,
                class_weight='balanced',
                verbose=-1,
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05,
                num_leaves=31, random_state=42,
                n_jobs=-1, verbose=-1,
            )

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self.metrics = self._compute_metrics(y_test, y_pred, X, y)
        return self.metrics


# ---------------------------------------------------------------------------
# Ensemble / AutoML-style pipeline
# ---------------------------------------------------------------------------

class EnsemblePipeline(MLPipeline):
    """
    Voting ensemble of Random Forest + Gradient Boosting (+ XGBoost if available).
    Best overall accuracy across most datasets.
    """

    def __init__(self, task_type: str = "classification"):
        super().__init__(task_type=task_type, model_name="Ensemble")

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        if isinstance(X, pd.DataFrame):
            X = self._preprocess_X(X, fit=True)

        stratify = None
        if self.task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            if all(c >= 2 for c in counts):
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        self.X_test = X_test
        self.y_test = y_test

        if self.task_type == "classification":
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')),
                ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ]
            if XGB_AVAILABLE:
                estimators.append(("xgb", xgb.XGBClassifier(
                    n_estimators=100,
                    eval_metric='logloss', random_state=42, n_jobs=-1,
                )))
            self.model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        else:
            estimators = [
                ("rf", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ]
            if XGB_AVAILABLE:
                estimators.append(("xgb", xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
            self.model = VotingRegressor(estimators=estimators, n_jobs=-1)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self.metrics = self._compute_metrics(y_test, y_pred, X, y)
        return self.metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Average feature importances from sub-estimators that support it."""
        importances = []
        estimators = self.model.estimators_
        for est in estimators:
            if hasattr(est, 'feature_importances_'):
                importances.append(est.feature_importances_)

        if not importances:
            return pd.DataFrame({"feature": self.feature_names, "importance": 0.0})

        avg_importance = np.mean(importances, axis=0)
        return pd.DataFrame({
            "feature": self.feature_names[:len(avg_importance)],
            "importance": avg_importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)