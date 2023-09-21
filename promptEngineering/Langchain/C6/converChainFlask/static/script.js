document.addEventListener('DOMContentLoaded', function () {
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const submitButton = document.getElementById('submit-button');

    // Function to handle form submission
    const handleSubmit = async () => {
        const inputText = userInput.value.trim();
        if (inputText === '') return;

        userInput.value = '';
        chatDisplay.innerHTML += `<p>User: ${inputText}</p>`;

        if (inputText.toLowerCase() === 'no') {
            chatDisplay.innerHTML += '<p>Goodbye!</p>';
            return;
        }

        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `input_text=${encodeURIComponent(inputText)}`
        }).then(response => response.json());

        chatDisplay.innerHTML += `<p>AI: ${response.response}</p>`;
    };

    // Handle form submission when the "Enter" key is pressed in the input field
    userInput.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent form submission via Enter key
            handleSubmit(); // Call the handleSubmit function
        }
    });

    // Handle form submission when the submit button is clicked
    submitButton.addEventListener('click', handleSubmit);
});
