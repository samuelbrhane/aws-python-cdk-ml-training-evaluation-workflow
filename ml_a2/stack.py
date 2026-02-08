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


        # Step 1: SageMaker Training job
        training_job_name = sfn.JsonPath.format(
            "ml-a2-train-{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        model_output_s3 = sfn.JsonPath.format(
            f"s3://{self.model_bucket.bucket_name}/models/" + "{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        train_task = tasks.SageMakerCreateTrainingJob(
            self,
            "TrainModel",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            training_job_name=training_job_name,
            role=self.sagemaker_execution_role,
            algorithm_specification=tasks.AlgorithmSpecification(
                training_image=tasks.DockerImage.from_registry(
                    "public.ecr.aws/sagemaker/sagemaker-xgboost:1.7-1"
                ),
                training_input_mode=tasks.InputMode.FILE,
            ),
            input_data_config=[
                tasks.Channel(
                    channel_name="train",
                    data_source=tasks.DataSource(
                        s3_data_source=tasks.S3DataSource(
                            s3_location=tasks.S3Location.from_uri(
                                f"s3://{self.training_data_bucket.bucket_name}/train/"
                            ),
                            s3_data_type=tasks.S3DataType.S3_PREFIX,
                        )
                    ),
                    content_type="text/csv",
                )
            ],
            output_data_config=tasks.OutputDataConfig(
                s3_output_location=tasks.S3Location.from_uri(model_output_s3)
            ),
            resource_config=tasks.ResourceConfig(
                instance_count=1,
                instance_type=ec2.InstanceType("ml.m5.large"),
                volume_size=Size.gibibytes(30),
            ),
            stopping_condition=tasks.StoppingCondition(max_runtime=Duration.hours(1)),
            hyperparameters={
                "max_depth": "5",
                "eta": "0.2",
                "objective": "binary:logistic",
                "num_round": "20",
            },
        )


        # Step 2: SageMaker Processing job (Evaluation)
        processing_job_name = sfn.JsonPath.format(
            "ml-a2-eval-{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        evaluation_output_s3 = sfn.JsonPath.format(
            f"s3://{self.evaluation_bucket.bucket_name}/evaluations/" + "{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        # This points at the model artifacts produced by training.
        model_artifacts_s3_prefix = sfn.JsonPath.format(
            f"s3://{self.model_bucket.bucket_name}/models/" + "{}",
            sfn.JsonPath.string_at("$$.Execution.Name"),
        )

        # Evaluation script placeholder location (upload after deploy)
        evaluation_script_s3 = f"s3://{self.training_data_bucket.bucket_name}/evaluation/evaluate.py"

        eval_task = tasks.SageMakerCreateProcessingJob(
            self,
            "EvaluateModel",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            processing_job_name=processing_job_name,
            role=self.sagemaker_execution_role,
            app_specification=tasks.AppSpecification(
                image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                container_entrypoint=["python3", "/opt/ml/processing/code/evaluate.py"],
            ),
            processing_resources=tasks.ProcessingResources(
                cluster_config=tasks.ClusterConfig(
                    instance_count=1,
                    instance_type=ec2.InstanceType("ml.m5.large"),
                    volume_size=Size.gibibytes(30),
                )
            ),
            processing_inputs=[
                
                # Pull evaluation code from S3 into /opt/ml/processing/code/
                tasks.ProcessingInput(
                    input_name="code",
                    source=tasks.ProcessingInputSource(
                        s3_input=tasks.S3Input(
                            s3_uri=evaluation_script_s3,
                            local_path="/opt/ml/processing/code",
                            s3_data_type=tasks.S3DataType.S3_PREFIX,
                            s3_input_mode=tasks.S3InputMode.FILE,
                        )
                    ),
                ),
                
                # Pull model artifacts output by training into /opt/ml/processing/model/
                tasks.ProcessingInput(
                    input_name="model",
                    source=tasks.ProcessingInputSource(
                        s3_input=tasks.S3Input(
                            s3_uri=model_artifacts_s3_prefix,
                            local_path="/opt/ml/processing/model",
                            s3_data_type=tasks.S3DataType.S3_PREFIX,
                            s3_input_mode=tasks.S3InputMode.FILE,
                        )
                    ),
                ),
                
                # allow the evaluator to read a validation set if you add it
                tasks.ProcessingInput(
                    input_name="validation",
                    source=tasks.ProcessingInputSource(
                        s3_input=tasks.S3Input(
                            s3_uri=f"s3://{self.training_data_bucket.bucket_name}/validation/",
                            local_path="/opt/ml/processing/validation",
                            s3_data_type=tasks.S3DataType.S3_PREFIX,
                            s3_input_mode=tasks.S3InputMode.FILE,
                        )
                    ),
                ),
            ],
            processing_outputs=[
                tasks.ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/output",
                    destination=tasks.S3Output(
                        s3_uri=evaluation_output_s3,
                        s3_upload_mode=tasks.S3UploadMode.END_OF_JOB,
                    ),
                )
            ],
        )
        
        
        # Reliability: retries + catch
        for task in (train_task, eval_task):
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


    
        # Step Functions definition: Training -> Evaluation
        definition = sfn.Chain.start(train_task).next(eval_task)

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