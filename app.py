#!/usr/bin/env python3
import aws_cdk as cdk

from ml_a2.stack import MlA2TrainingEvaluationStack

app = cdk.App()

MlA2TrainingEvaluationStack(
    app,
    "MlA2TrainingEvaluationStack",
)

app.synth()
