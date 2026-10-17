\# BusPredict Testing Report



\## 1. Introduction



This report describes the planned testing of the AI-Based Smart Vehicle Maintenance and Inspection System (BusPredict).



\## 2. Testing Objectives



\- Check valid vehicle inputs.

\- Check missing and invalid inputs.

\- Check the prediction feature.

\- Record errors and limitations.



\## 3. Test Cases



| Test Case | Action | Expected Result | Actual Result |

|---|---|---|---|

| Valid vehicle input | Submit valid vehicle information | A prediction is returned | Not tested |

| Missing input | Leave a required field empty | The application handles the missing input appropriately | Not tested |

| Invalid numeric input | Submit a negative value for reported issues | The API rejects the invalid value | Not tested |

| Prediction request | Submit valid inspection data | The system returns a prediction or an error message | Not tested |



\## 4. Limitations



The prediction results depend on the quality of the available data and the trained model. Further testing is needed to evaluate accuracy and reliability.



\## 5. Conclusion



This report provides a starting point for testing BusPredict. Actual results will be recorded after the tests are performed.

