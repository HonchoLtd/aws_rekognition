package text_detection

import (
	"context"
	"errors"
	"fmt"
	_ "image/png" // enable PNG decoding as well
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

type TextDetection interface {
	DetectText(ctx context.Context, image []byte) ([]string, error)
}

type rekognitionTextDetector struct {
	client *rekognition.Client
}

func NewRekognitionTextDetector(client *rekognition.Client) TextDetection {
	return &rekognitionTextDetector{client: client}
}

// DetectText implementation of TextDetection using AWS Rekognition.
func (r *rekognitionTextDetector) DetectText(ctx context.Context, inputImage []byte) ([]string, error) {
	log.Printf("Initiate Text Detection Services")
	if len(inputImage) == 0 {
		return nil, errors.New("input image is empty")
	}
	log.Printf("Start Text Detection Services")

	// Build Rekognition request
	out, err := r.client.DetectText(ctx, &rekognition.DetectTextInput{
		Image: &types.Image{
			Bytes: inputImage,
		},
	})
	log.Printf("Finish Text Detection Services")

	if err != nil {
		return nil, fmt.Errorf("rekognition DetectText failed: %w", err)
	}

	var texts []string
	log.Printf("Load Text with Word Type")
	for _, td := range out.TextDetections {
		// We only care about WORD type, not LINE
		if td.Type != types.TextTypesWord {
			continue
		}
		if td.DetectedText == nil {
			continue
		}

		const minConfidence = 70.0
		if td.Confidence != nil && *td.Confidence < minConfidence {
			continue
		}
		texts = append(texts, aws.ToString(td.DetectedText))
	}
	log.Printf("Return Text Detection Results")

	return texts, nil
}
