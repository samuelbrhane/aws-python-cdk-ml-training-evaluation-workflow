import aws_cdk as cdk
from aws_cdk.assertions import Template

from ml_a2.stack import MlA2TrainingEvaluationStack


def test_core_resources_created():
    app = cdk.App()
    stack = MlA2TrainingEvaluationStack(app, "TestMlA2Stack")
    template = Template.from_stack(stack)

    # S3 buckets: training data, model artifacts, evaluation outputs
    template.resource_count_is("AWS::S3::Bucket", 3)

    # Step Functions State Machine
    template.resource_count_is("AWS::StepFunctions::StateMachine", 1)

    # SNS topic for failure alerts
    template.resource_count_is("AWS::SNS::Topic", 1)

    # CloudWatch alarm for failed executions
    template.resource_count_is("AWS::CloudWatch::Alarm", 1)

    # CloudWatch LogGroup for Step Functions logging
    template.resource_count_is("AWS::Logs::LogGroup", 1)

    # IAM role for SageMaker
    template.resource_count_is("AWS::IAM::Role", 3)
