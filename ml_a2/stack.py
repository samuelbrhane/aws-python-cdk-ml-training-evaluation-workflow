from __future__ import annotations

from constructs import Construct
from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    Size,
    aws_s3 as s3,
    aws_sns as sns,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_stepfunctions_tasks as tasks,
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
)


class MlA2TrainingEvaluationStack(Stack):
    """
    ML-A2 — Training + Evaluation Workflow

    Flow (minimum scope):
      1) SageMaker Training job
      2) SageMaker Processing job for evaluation
      3) Step Functions orchestration + failure alerts
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)


        # S3 buckets for data
        self.training_data_bucket = s3.Bucket(
            self,
            "TrainingDataBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # S3 model artifacts
        self.model_bucket = s3.Bucket(
            self,
            "ModelArtifactsBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # S3 evaluation outputs 
        self.evaluation_bucket = s3.Bucket(
            self,
            "EvaluationBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        
        
        # SNS topic for failure alerts
        self.failure_topic = sns.Topic(
            self,
            "FailureAlertsTopic",
            display_name="ML-A2 Training/Evaluation Failure Alerts",
        )

        # CloudWatch Logs (Step Functions)
        self.sfn_log_group = logs.LogGroup(
            self,
            "StateMachineLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # IAM role for SageMaker jobs
        self.sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        

        # Allow SageMaker to read training data and write model/eval outputs
        self.training_data_bucket.grant_read(self.sagemaker_execution_role)
        self.model_bucket.grant_read_write(self.sagemaker_execution_role)
        self.evaluation_bucket.grant_read_write(self.sagemaker_execution_role)
        
        
        
        # Step Functions: failure notification task
        notify_failure = tasks.SnsPublish(
            self,
            "NotifyFailure",
            topic=self.failure_topic,
            subject="ML-A2 Training + Evaluation Pipeline Failed",
            message=sfn.TaskInput.from_json_path_at("$"),
        )
