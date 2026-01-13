import logging
import os
from datetime import datetime
from typing import ClassVar, Optional, Tuple

import boto3
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from pydantic import BaseModel, Field

from aidial_adapter_anthropic._utils.concurrency import make_async
from aidial_adapter_anthropic._utils.env import get_aws_default_region

_log = logging.getLogger(__name__)

_UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"


class AWSClientCredentialArgs(BaseModel):
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None


class AWSClientCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None = None

    def get_credentials(
        self,
    ) -> Tuple[datetime | None, AWSClientCredentialArgs]:
        return None, AWSClientCredentialArgs(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


class AWSAssumeRoleCredentials(BaseModel):
    aws_assume_role_arn: str

    async def get_credentials(
        self, region: str
    ) -> Tuple[datetime, AWSClientCredentialArgs]:
        sts_client = await make_async(
            lambda: boto3.Session().client("sts", region_name=region)
        )

        response = sts_client.assume_role(
            RoleArn=self.aws_assume_role_arn,
            RoleSessionName="BedrockAccessSession",
        )

        creds = response["Credentials"]

        return creds["Expiration"], AWSClientCredentialArgs(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )


class AWSUpstreamConfig(BaseModel):
    region: str
    credentials: AWSClientCredentials | AWSAssumeRoleCredentials | None = None

    @classmethod
    async def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> "AWSUpstreamConfig":
        conf = request.headers.get(_UPSTREAM_CONFIG_HEADER_NAME)
        upstream_config = (
            UpstreamConfigData.parse_raw(conf) if conf else UpstreamConfigData()
        )

        return cls(
            region=upstream_config.region,
            credentials=upstream_config._get_client_credentials(),
        )

    async def get_credentials(
        self,
    ) -> Tuple[datetime | None, AWSClientCredentialArgs]:
        if self.credentials is None:
            return (None, AWSClientCredentialArgs())
        if isinstance(self.credentials, AWSClientCredentials):
            return self.credentials.get_credentials()
        return await self.credentials.get_credentials(self.region)


class ApiKeyUpstreamConfig(BaseModel):
    _UPSTREAM_API_KEY_HEADER_NAME: ClassVar[str] = "x-upstream-key"

    api_key: str

    @classmethod
    def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> Optional["ApiKeyUpstreamConfig"]:
        key = request.headers.get(cls._UPSTREAM_API_KEY_HEADER_NAME)
        return None if key is None else cls(api_key=key)


UpstreamConfig = ApiKeyUpstreamConfig | AWSUpstreamConfig


async def parse_upstream_config(
    request: FromRequestDeploymentMixin,
) -> UpstreamConfig:
    if (conf := ApiKeyUpstreamConfig.from_request(request)) is not None:
        _log.debug("accessing deployment via platform api-key")
        return conf

    _log.debug("accessing deployment via cloud creds")
    return await AWSUpstreamConfig.from_request(request)


class UpstreamConfigData(BaseModel):
    region: str = Field(default_factory=get_aws_default_region)
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = os.getenv("AWS_SESSION_TOKEN")
    aws_assume_role_arn: str | None = os.getenv("AWS_ASSUME_ROLE_ARN")

    def _get_client_credentials(
        self,
    ) -> AWSClientCredentials | AWSAssumeRoleCredentials | None:

        if self.aws_access_key_id and self.aws_secret_access_key:
            return AWSClientCredentials(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
            )

        if self.aws_assume_role_arn:
            return AWSAssumeRoleCredentials(
                aws_assume_role_arn=self.aws_assume_role_arn
            )

        return None
