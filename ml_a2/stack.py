from __future__ import annotations

from constructs import Construct
from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    aws_s3 as s3,
    aws_sns as sns,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_iam as iam,
    aws_stepfunctions_tasks as tasks,
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
)


class MlA2TrainingEvaluationStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 Buckets
        self.training_data_bucket = s3.Bucket(
            self,
            "TrainingDataBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            # Change to RETAIN for production
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        self.model_bucket = s3.Bucket(
            self,
            "ModelArtifactsBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        self.evaluation_bucket = s3.Bucket(
            self,
            "EvaluationBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # SNS Topic for failure alerts
        self.failure_topic = sns.Topic(
            self,
            "FailureAlertsTopic",
            display_name="ML-A2 Training/Evaluation Failure Alerts",
        )

        # CloudWatch Log Group for Step Functions
        self.sfn_log_group = logs.LogGroup(
            self,
            "StateMachineLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # IAM Role for SageMaker (training + processing)
        self.sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )

        self.training_data_bucket.grant_read(self.sagemaker_execution_role)
        self.model_bucket.grant_read_write(self.sagemaker_execution_role)
        self.evaluation_bucket.grant_read_write(self.sagemaker_execution_role)

        # Shared: SNS failure notification state
        notify_failure = tasks.SnsPublish(
            self,
            "NotifyFailure",
            topic=self.failure_topic,
            subject="ML-A2 Training + Evaluation Pipeline Failed",
            message=sfn.TaskInput.from_json_path_at("$"),
        )

        # S3 paths
        training_input_s3 = f"s3://{self.training_data_bucket.bucket_name}/train/"
        model_output_s3 = f"s3://{self.model_bucket.bucket_name}/models/"
        model_artifacts_s3_prefix = f"s3://{self.model_bucket.bucket_name}/models/"
        evaluation_output_s3 = f"s3://{self.evaluation_bucket.bucket_name}/evaluations/"
        evaluation_code_s3_prefix = f"s3://{self.training_data_bucket.bucket_name}/evaluation/"
        validation_s3_prefix = f"s3://{self.training_data_bucket.bucket_name}/validation/"

        # Dynamic job names (unique per execution)
        training_job_name = sfn.JsonPath.format(
            "ml-a2-train-{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        processing_job_name = sfn.JsonPath.format(
            "ml-a2-eval-{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        # Step 1: SageMaker Training Job
        create_training_job = tasks.CallAwsService(
            self,
            "CreateTrainingJob",
            service="sagemaker",
            action="createTrainingJob",
            iam_resources=["*"],
            parameters={
                "TrainingJobName": training_job_name,
                "AlgorithmSpecification": {
                    "TrainingImage": "public.ecr.aws/sagemaker/sagemaker-xgboost:1.7-1",
                    "TrainingInputMode": "File",
                },
                "RoleArn": self.sagemaker_execution_role.role_arn,
                "InputDataConfig": [
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": training_input_s3,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "ContentType": "text/csv",
                    }
                ],
                "OutputDataConfig": {"S3OutputPath": model_output_s3},
                "ResourceConfig": {
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30,
                },
                "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
                "HyperParameters": {
                    "max_depth": "5",
                    "eta": "0.2",
                    "objective": "binary:logistic",
                    "num_round": "20",
                },
            },
            result_path="$.training.create",
        )

        describe_training_job = tasks.CallAwsService(
            self,
            "DescribeTrainingJob",
            service="sagemaker",
            action="describeTrainingJob",
            iam_resources=["*"],
            parameters={"TrainingJobName": training_job_name},
            result_path="$.training.describe",
        )

        wait_training = sfn.Wait(
            self,
            "WaitTraining",
            time=sfn.WaitTime.duration(Duration.seconds(30)),
        )

        training_status_choice = sfn.Choice(self, "TrainingStatus?")

        # Step 2: SageMaker Processing Job (Evaluation)
        create_processing_job = tasks.CallAwsService(
            self,
            "CreateProcessingJob",
            service="sagemaker",
            action="createProcessingJob",
            iam_resources=["*"],
            parameters={
                "ProcessingJobName": processing_job_name,
                "RoleArn": self.sagemaker_execution_role.role_arn,
                "AppSpecification": {
                    
                    # NOTE: region-specific image URI (us-east-1). Adjust for your region.
                    "ImageUri": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                    "ContainerEntrypoint": [
                        "python3",
                        "/opt/ml/processing/code/evaluate.py",
                    ],
                },
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.large",
                        "VolumeSizeInGB": 30,
                    }
                },
                "ProcessingInputs": [
                    {
                        "InputName": "code",
                        "S3Input": {
                            "S3Uri": evaluation_code_s3_prefix,
                            "LocalPath": "/opt/ml/processing/code",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File",
                        },
                    },
                    {
                        "InputName": "model",
                        "S3Input": {
                            "S3Uri": model_artifacts_s3_prefix,
                            "LocalPath": "/opt/ml/processing/model",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File",
                        },
                    },
                    {
                        "InputName": "validation",
                        "S3Input": {
                            "S3Uri": validation_s3_prefix,
                            "LocalPath": "/opt/ml/processing/validation",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File",
                        },
                    },
                ],
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "evaluation",
                            "S3Output": {
                                "S3Uri": evaluation_output_s3,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
            },
            result_path="$.evaluation.create",
        )

        describe_processing_job = tasks.CallAwsService(
            self,
            "DescribeProcessingJob",
            service="sagemaker",
            action="describeProcessingJob",
            iam_resources=["*"],
            parameters={"ProcessingJobName": processing_job_name},
            result_path="$.evaluation.describe",
        )

        wait_evaluation = sfn.Wait(
            self,
            "WaitEvaluation",
            time=sfn.WaitTime.duration(Duration.seconds(30)),
        )

        evaluation_status_choice = sfn.Choice(self, "EvaluationStatus?")
        pipeline_succeeded = sfn.Succeed(self, "PipelineSucceeded")

        # Retries + Catch on both create-job tasks
        for task in (create_training_job, create_processing_job):
            task.add_retry(
                errors=[
                    "SageMaker.AmazonSageMakerException",
                    "SageMaker.ResourceLimitExceeded",
                    "SageMaker.ThrottlingException",
                    "States.Timeout",
                    "States.TaskFailed",
                ],
                interval=Duration.seconds(30),
                backoff_rate=2.0,
                max_attempts=3,
            )
            task.add_catch(
                handler=notify_failure,
                errors=["States.ALL"],
                result_path="$.error",
            )

        # Wire evaluation polling loop
        evaluation_flow = create_processing_job.next(
            describe_processing_job.next(
                evaluation_status_choice
                .when(
                    sfn.Condition.string_equals(
                        "$.evaluation.describe.ProcessingJobStatus",
                        "Completed",
                    ),
                    pipeline_succeeded,
                )
                .when(
                    sfn.Condition.or_(
                        sfn.Condition.string_equals(
                            "$.evaluation.describe.ProcessingJobStatus",
                            "Failed",
                        ),
                        sfn.Condition.string_equals(
                            "$.evaluation.describe.ProcessingJobStatus",
                            "Stopped",
                        ),
                    ),
                    notify_failure,
                )
                .otherwise(wait_evaluation.next(describe_processing_job))
            )
        )

        train_flow = create_training_job.next(
            describe_training_job.next(
                training_status_choice
                .when(
                    sfn.Condition.string_equals(
                        "$.training.describe.TrainingJobStatus",
                        "Completed",
                    ),
                    evaluation_flow,
                )
                .when(
                    sfn.Condition.or_(
                        sfn.Condition.string_equals(
                            "$.training.describe.TrainingJobStatus",
                            "Failed",
                        ),
                        sfn.Condition.string_equals(
                            "$.training.describe.TrainingJobStatus",
                            "Stopped",
                        ),
                    ),
                    notify_failure,
                )
                .otherwise(wait_training.next(describe_training_job))
            )
        )

        # State Machine definition starts from train_flow
        definition = sfn.Chain.start(train_flow)

        self.state_machine = sfn.StateMachine(
            self,
            "TrainingEvaluationStateMachine",
            definition=definition,
            timeout=Duration.hours(2),
            logs=sfn.LogOptions(
                destination=self.sfn_log_group,
                level=sfn.LogLevel.ALL,
            ),
            tracing_enabled=True,
        )

        # CloudWatch Alarm on failed executions
        self.failed_executions_alarm = cw.Alarm(
            self,
            "StateMachineFailedExecutionsAlarm",
            metric=self.state_machine.metric_failed(),
            threshold=1,
            evaluation_periods=1,
            datapoints_to_alarm=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="Alarm when the ML-A2 training/evaluation Step Functions execution fails.",
        )

        self.failed_executions_alarm.add_alarm_action(
            cw_actions.SnsAction(self.failure_topic)
        )