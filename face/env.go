package face

type Env struct {
	AwsRegion          string `env:"AWS_REGION" json:"AWS_REGION"`
	AwsAccessKeyID     string `env:"AWS_ACCESS_KEY_ID" json:"AWS_ACCESS_KEY_ID"`
	AwsSecretAccessKey string `env:"AWS_SECRET_ACCESS_KEY" json:"AWS_SECRET_ACCESS_KEY"`
	AwsBucketName      string `env:"AWS_BUCKET_NAME" json:"AWS_BUCKET_NAME"`
}
