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
