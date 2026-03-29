#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)
    logger.info("Downloaded dataset: %d rows, %d columns", *df.shape)

    # Remove price outliers
    logger.info("Filtering price between %.2f and %.2f", args.min_price, args.max_price)
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info("Rows after price filter: %d", len(df))

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])
    logger.info("Converted last_review to datetime")

    # Remove properties outside NYC proper boundaries
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info("Rows after geolocation filter: %d", len(df))

    df.to_csv("clean_sample.csv", index=False)
    logger.info("Saved clean_sample.csv")

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("Uploaded artifact %s", args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name of the input artifact in W&B",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact to create in W&B",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to include (filter out lower outliers)",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to include (filter out upper outliers)",
        required=True,
    )

    args = parser.parse_args()

    go(args)
