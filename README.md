# Furniture Layout Planner

An AI-powered furniture layout optimization tool that uses fuzzy logic to create optimal room layouts based on furniture relationships, ergonomics, and design principles.

## Features

- **Intelligent Furniture Placement**: Optimizes furniture arrangements based on spatial relationships and design principles
- **Fuzzy Logic Evaluation**: Uses fuzzy logic to evaluate layout quality based on multiple design criteria
- **Room Zoning**: Automatically divides rooms into functional zones (entertainment, dining, storage)
- **Multiple Layout Generation**: Creates several alternative layouts for comparison
- **Relationship-Aware Placement**: Understands relationships between furniture (e.g., sofas face TV, coffee tables near sofas)
- **Customizable Parameters**: Adjust room dimensions, furniture sizes, and design preferences

## Project Structure

```
furniture_planner/
├── core/                   # Core implementation
│   ├── furniture.py        # Enhanced furniture class
│   ├── room.py             # Room environment
│   ├── fuzzy_logic.py      # Fuzzy logic implementation
│   └── optimizer.py        # Layout optimization
├── layout/                 # Layout algorithms
│   ├── entertainment.py    # Entertainment zone layout
│   ├── dining.py           # Dining zone layout
│   ├── storage.py          # Storage zone layout
│   └── pathway.py          # Pathway generation
├── utils/                  # Utility functions
│   ├── visualization.py    # Enhanced visualization
│   └── metrics.py          # Layout evaluation metrics
├── app/                    # User interface
│   └── streamlit_app.py    # Streamlit interface
├── requirements.txt        # Project dependencies
└── main.py                 # Entry point
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/furniture-planner.git
cd furniture-planner
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Running the Demo

To run a demonstration of the optimizer:

```bash
python main.py --demo
```

This will create a sample room, optimize furniture placement, and generate visualizations of the layout.

### Using the Streamlit App

For a user-friendly interface:

```bash
streamlit run app/streamlit_app.py
```

## Implementation Details

### Furniture Relationships

The system models relationships between furniture pieces:

- **TV and Sofa**: Sofas face the TV with optimal viewing distance of 8 feet
- **Coffee Table**: Positioned between sofas and TV at about 2 feet from the sofa
- **Dining Chairs**: Positioned around the dining table and facing it
- **Bookshelves**: Prefer to be against walls or in corners
- **Side Tables**: Positioned near sofas
- **Lamps**: Positioned near sofas or side tables
- **Plants**: Prefer to be near windows or in corners

### Fuzzy Logic Evaluation

The system uses fuzzy logic to evaluate layout quality based on:

- **Distance Relationships**: Optimal distances between related furniture
- **Facing Relationships**: Whether furniture pieces face related pieces appropriately
- **Wall Preferences**: Whether furniture that prefers walls is against walls
- **Corner Preferences**: Whether furniture that prefers corners is in corners
- **Window Relationships**: Whether furniture that prefers/avoids windows is appropriately placed
- **Zone Coherence**: Whether furniture is in appropriate functional zones
