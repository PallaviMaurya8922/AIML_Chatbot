// Function to handle feedback submission
function submitFeedback(button, feedbackType) {
    const responseId = button.getAttribute("data-id");  // Get the ObjectId from the button's data-id attribute
    fetch(`/feedback/${responseId}/${feedbackType}`, {
        method: "POST"
    }).then(response => {
        if (response.ok) {
            alert("Thank you for your feedback!");
        } else {
            alert("Error submitting feedback.");
        }
    });
}

// Validate input to prevent empty submissions
function validateInput() {
    const queryInput = document.getElementById("query");
    const appendMessageTo = document.getElementById("query-form")
    const errorElement = document.querySelector('.error-message');
    if (!queryInput.value.trim()) {
        if (!errorElement) {
            const errorMessage = document.createElement("span");
            errorMessage.classList.add('error-message');
            errorMessage.textContent = "Please enter a valid text.";
            errorMessage.style.color = "red";
            errorMessage.style.paddingLeft = "2px";
            errorMessage.style.marginTop = "10px"; // Optional: Make the error text bold
            appendMessageTo.parentNode.appendChild(errorMessage);
        }
        return false
    }
    return true; // Allow form submission
}
function submitQuery() {
    if (!validateInput()) {
        event.preventDefault();
    }
}