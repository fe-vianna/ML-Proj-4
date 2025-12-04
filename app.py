from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

NUMERIC_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]

CATEGORICAL_FEATURES = [
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "issue_d",
    "purpose",
    "addr_state",
    "earliest_cr_line",
    "revol_util",
    "initial_list_status",
    "application_type",
]

FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@app.route("/")
def index():
    return render_template(
        "index.html",
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        data = {}

        # ---------- GET ----------
        if request.method == "GET":
            for name in FEATURE_NAMES:
                value = request.args.get(name, None)
                if value is None:
                    return jsonify(
                        {"error": f"Missing parameter '{name}' in query string."}
                    ), 400

                if name in NUMERIC_FEATURES:
                    data[name] = float(value)
                else:  # categorical
                    data[name] = str(value)

        # ---------- POST ----------
        else:
            if request.is_json:
                req_json = request.get_json()
            else:
                req_json = request.form

            for name in FEATURE_NAMES:
                value = req_json.get(name, None)

                if value is None or value == "":
                    if name in NUMERIC_FEATURES:
                        data[name] = np.nan
                    else:
                        data[name] = None
                else:
                    if name in NUMERIC_FEATURES:
                        data[name] = float(value)
                    else:
                        data[name] = str(value)

        X = pd.DataFrame([data], columns=FEATURE_NAMES)


        # ---------- Predict ----------
        raw_pred = model.predict(X)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            confidence = float(np.max(probs))

        return jsonify(
            {
                "inputs": data,
                "prediction": str(raw_pred),
                "confidence": confidence,
            }
        )

    except Exception as e:
        error_payload = {"error": str(e)}
        try:
            error_payload["dtypes"] = X.dtypes.astype(str).to_dict()
        except Exception:
            pass
        return jsonify(error_payload), 500


if __name__ == "__main__":
    app.run(debug=True)