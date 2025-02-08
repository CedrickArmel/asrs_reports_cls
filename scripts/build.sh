#!/bin/bash
usage() {
    echo "Usage: $0 -p <project> -r <region> -t <file> -b <branch> [-s <substitutions>]"
    exit 1
}

while getopts "p:r:t:b:s" opt; do
    case $opt in
        p) project="$OPTARG";;
        r) region="$OPTARG";;
        t) trigger="$OPTARG";;
        b) branch="$OPTARG";;
        s) substitutions="$OPTARG";;
        *) usage;;
    esac
done

if [ -z "$project" ] || [ -z "$region" ] || [ -z "$trigger" ] || [ -z "$branch" ]; then
    usage
fi

export CLOUDSDK_CORE_DISABLE_PROMPTS=1

gcloud config set project "$project"

if [ -z "$substitutions" ]; then
    BUILD_ID=$(gcloud beta builds triggers run "$trigger" --region="$region" --branch="$branch" --quiet --format="value(metadata.build.id)")
else
    BUILD_ID=$(gcloud beta builds triggers run "$trigger" --region="$region" --substitutions="$substitutions" --branch="$branch" --quiet --format="value(metadata.build.id)")
fi

echo "BUILD_ID = $BUILD_ID"
gcloud beta builds log --stream "$BUILD_ID" --region="$region"
BUILD_STATUS=$(gcloud beta builds describe "$BUILD_ID" --format="value(status)" --region="$region")
echo "Build status = $BUILD_STATUS"

if [ "$BUILD_STATUS" == "SUCCESS" ]; then
    exit 0
else
    exit 1
fi