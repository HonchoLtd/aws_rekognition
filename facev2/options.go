package facev2

// This file holds the functional options for SearchFacesByFaceId, the only
// grouping-related Face method left after CreateUser/AssociateFaces/
// SearchUsersByFaceId were removed in favor of a locally-managed user_id
// (see facev2/simulation.DBGrouper and facev2/store). It falls back to AWS
// Rekognition's own defaults when every option is omitted.

// searchFacesConfig holds the optional parameters for SearchFacesByFaceId.
type searchFacesConfig struct {
	faceMatchThreshold *float32
	maxFaces           *int32
}

// SearchFacesOption configures a SearchFacesByFaceId call.
type SearchFacesOption func(*searchFacesConfig)

// WithSearchFacesMatchThreshold sets the minimum confidence (0-100) in the
// face match to return. AWS defaults to 80 when omitted.
func WithSearchFacesMatchThreshold(threshold float32) SearchFacesOption {
	return func(c *searchFacesConfig) { c.faceMatchThreshold = &threshold }
}

// WithSearchFacesMaxResults caps the number of matched faces returned.
func WithSearchFacesMaxResults(max int32) SearchFacesOption {
	return func(c *searchFacesConfig) { c.maxFaces = &max }
}
