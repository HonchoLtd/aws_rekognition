# Prerequisite:

```
go get github.com/aws/aws-sdk-go-v2/aws
go get github.com/aws/aws-sdk-go-v2/service/rekognition
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/samber/lo
```

Call This in converter to index face from image upload, imageID that will be return in `SearchFace` So make sure put correct id so you can easy find in gallery later on
```
IndexFace(ctx context.Context, image []byte, imageID string, eventID string) error
```

Call this in selfie to get face from selfie and search image in from eventID
```
SearchFace(ctx context.Context, imageSelfie []byte, eventID string) ([]string, error)
```

Make sure the eventID is same, or we can't make correct collections.

