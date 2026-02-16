import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.proportion import proportions_ztest


class SmartABTester:

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    # --------------------------------------
    # SRM CHECK (Sample Ratio Mismatch)
    # --------------------------------------
    def check_srm(self, df, variant_col="variant"):

        counts = df[variant_col].value_counts().values
        expected = np.repeat(np.mean(counts), len(counts))

        chi2 = ((counts - expected) ** 2 / expected).sum()
        pvalue = 1 - stats.chi2.cdf(chi2, df=len(counts)-1)

        return {
            "chi2": chi2,
            "pvalue": pvalue,
            "srm_detected": pvalue < self.alpha
        }

    # --------------------------------------
    # NORMALITY CHECK (Shapiro-Wilk)
    # --------------------------------------
    def _check_normality(self, data):

        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)

        stat, pvalue = shapiro(data)
        return pvalue > self.alpha

    # --------------------------------------
    # PROPORTION TEST (Binary)
    # --------------------------------------
    def proportion_test(self, control, treatment):

        successes = np.array([treatment.sum(), control.sum()])
        nobs = np.array([len(treatment), len(control)])

        stat, pvalue = proportions_ztest(successes, nobs)

        return {
            "test": "Proportion Z-Test",
            "statistic": stat,
            "pvalue": pvalue,
            "significant": pvalue < self.alpha
        }

    # --------------------------------------
    # TWO SAMPLE T-TEST
    # --------------------------------------
    def two_sample_ttest(self, control, treatment):

        stat, pvalue = stats.ttest_ind(
            treatment,
            control,
            equal_var=False
        )

        return {
            "test": "Two Sample T-Test",
            "statistic": stat,
            "pvalue": pvalue,
            "significant": pvalue < self.alpha
        }

    # --------------------------------------
    # MANN WHITNEY U TEST
    # --------------------------------------
    def mann_whitney_u_test(self, control, treatment):

        stat, pvalue = stats.mannwhitneyu(
            treatment,
            control,
            alternative="two-sided"
        )

        return {
            "test": "Mann-Whitney U",
            "statistic": stat,
            "pvalue": pvalue,
            "significant": pvalue < self.alpha
        }

    # --------------------------------------
    # MAIN DECISION ENGINE
    # --------------------------------------
    def analyze_test(self, df, metric_col, variant_col="variant"):

        # 1️⃣ SRM CHECK
        srm_result = self.check_srm(df, variant_col)

        variants = df[variant_col].unique()
        control = df[df[variant_col] == variants[0]][metric_col].dropna().values
        treatment = df[df[variant_col] == variants[1]][metric_col].dropna().values

        unique_vals = set(df[metric_col].dropna().unique())

        # 2️⃣ AUTO METRIC TYPE
        if unique_vals <= {0, 1}:
            result = self.proportion_test(control, treatment)
            result["metric_type"] = "binary"
        else:
            is_normal_control = self._check_normality(control)
            is_normal_treatment = self._check_normality(treatment)

            if is_normal_control and is_normal_treatment:
                result = self.two_sample_ttest(control, treatment)
                result["metric_type"] = "continuous_normal"
            else:
                result = self.mann_whitney_u_test(control, treatment)
                result["metric_type"] = "continuous_non_normal"

        result["srm"] = srm_result

        return result
