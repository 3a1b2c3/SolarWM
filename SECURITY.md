# Security

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**. That opens a
draft advisory visible only to the maintainers.

**Do not open a public issue**, and do not put credentials, tokens, or internal
hostnames in a report — a report is a document, and anything pasted into one is
as exposed as the thing it describes. Include the affected version or commit,
what an attacker gains, and the smallest reproduction you have.

Expect an acknowledgement within a week. Fixes ship in a normal release; we will
say so in the advisory when a report does not warrant one.

## Scope

This is a training and inference framework, not a hosted service. The things
worth reporting are the ones that cross a trust boundary: code execution while
loading a config, checkpoint, or dataset shard; path traversal out of an output
or extraction root; a credential reaching a persisted artifact; or an integrity
check that can be bypassed.

Model outputs are not in scope. A generated video that is wrong, offensive, or
unlike its prompt is a model-quality question, not a vulnerability.

## What the framework already does

Resolved run configs and launch manifests are persisted as artifacts, so treat a
config as something that will be written down. Keys literally named `token`,
`secret`, `password`, `api_key` or `access_token` are rejected, but that check is
a name match on those five and nothing more: it does not inspect values, does not
catch `aws_secret_access_key`, `client_secret`, `hf_token` or `private_key`, and
runs **after** `${VAR}` expansion — so a credential injected through the
environment is expanded and then persisted. Do not put credentials in a config.
Use workload identity, or a credential-file reference outside the repository. Published indexes may
contain only relative POSIX shard keys. Tar members are read without filesystem
extraction, and object-store shards are generation-pinned and integrity-checked
before use.
