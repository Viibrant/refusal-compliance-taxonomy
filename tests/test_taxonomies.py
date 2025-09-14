"""Tests for the taxonomies module."""

import pytest
from rejection_detection.taxonomies import (
    OutcomeType,
    RefusalStyle,
    ComplianceStyle,
    HarmCategory,
    HarmlessCategory,
    HeadConfig,
    get_head_configs,
    get_head_config,
    get_label_to_id_mapping,
    get_id_to_label_mapping,
    get_all_head_configs,
    is_refusal_label,
    is_compliance_label,
    get_outcome_precedence,
    get_outcome_definitions,
    get_minimal_examples,
    get_harmless_category_definitions
)


class TestEnums:
    """Test enum definitions."""
    
    def test_outcome_type_enum(self):
        """Test OutcomeType enum values."""
        assert OutcomeType.REFUSAL_DIRECT.value == "REFUSAL.DIRECT"
        assert OutcomeType.REFUSAL_SOFT.value == "REFUSAL.SOFT"
        assert OutcomeType.REFUSAL_PARTIAL.value == "REFUSAL.PARTIAL"
        assert OutcomeType.REFUSAL_CAPABILITY.value == "REFUSAL.CAPABILITY"
        assert OutcomeType.REFUSAL_OVER.value == "REFUSAL.OVER"
        assert OutcomeType.COMPLY_BENIGN.value == "COMPLY.BENIGN"
        assert OutcomeType.COMPLY_TRANSFORM.value == "COMPLY.TRANSFORM"
        assert OutcomeType.COMPLY_UNSAFE.value == "COMPLY.UNSAFE"
        
        # Test new compliance types
        assert OutcomeType.COMPLY_CONDITIONAL.value == "COMPLY.CONDITIONAL"
        assert OutcomeType.COMPLY_EDUCATIONAL.value == "COMPLY.EDUCATIONAL"
        assert OutcomeType.COMPLY_REDIRECTED.value == "COMPLY.REDIRECTED"
        assert OutcomeType.COMPLY_PARTIAL_SAFE.value == "COMPLY.PARTIAL_SAFE"
        
        # Test all values
        all_values = [e.value for e in OutcomeType]
        assert len(all_values) == 12  # Updated count
        assert "REFUSAL.DIRECT" in all_values
        assert "COMPLY.BENIGN" in all_values
        assert "COMPLY.CONDITIONAL" in all_values
        assert "COMPLY.EDUCATIONAL" in all_values
        assert "COMPLY.REDIRECTED" in all_values
        assert "COMPLY.PARTIAL_SAFE" in all_values
    
    def test_refusal_style_enum(self):
        """Test RefusalStyle enum values."""
        assert RefusalStyle.STYLE_DIRECT.value == "STYLE.DIRECT"
        assert RefusalStyle.STYLE_EXPLAIN.value == "STYLE.EXPLAIN"
        assert RefusalStyle.STYLE_REDIRECT.value == "STYLE.REDIRECT"
        assert RefusalStyle.STYLE_CLARIFY.value == "STYLE.CLARIFY"
        assert RefusalStyle.STYLE_APPEAL_TO_VALUES.value == "STYLE.APPEAL_TO_VALUES"
        assert RefusalStyle.STYLE_EMPATHETIC.value == "STYLE.EMPATHETIC"
        assert RefusalStyle.STYLE_RATE_LIMIT.value == "STYLE.RATE_LIMIT"
        
        # Test all values
        all_values = [e.value for e in RefusalStyle]
        assert len(all_values) == 7
    
    def test_compliance_style_enum(self):
        """Test ComplianceStyle enum values."""
        assert ComplianceStyle.STYLE_DIRECT_ANSWER.value == "STYLE.DIRECT_ANSWER"
        assert ComplianceStyle.STYLE_HIGH_LEVEL.value == "STYLE.HIGH_LEVEL"
        assert ComplianceStyle.STYLE_STEP_BY_STEP.value == "STYLE.STEP_BY_STEP"
        assert ComplianceStyle.STYLE_WITH_WARNINGS.value == "STYLE.WITH_WARNINGS"
        assert ComplianceStyle.STYLE_REFRACT.value == "STYLE.REFRACT"
        assert ComplianceStyle.STYLE_CLARIFY_THEN_COMPLY.value == "STYLE.CLARIFY_THEN_COMPLY"
        assert ComplianceStyle.STYLE_TOOL_USE.value == "STYLE.TOOL_USE"
        assert ComplianceStyle.STYLE_CITED.value == "STYLE.CITED"
        assert ComplianceStyle.STYLE_CREATIVE.value == "STYLE.CREATIVE"
        assert ComplianceStyle.STYLE_TRANSFORM_ONLY.value == "STYLE.TRANSFORM_ONLY"
        assert ComplianceStyle.STYLE_PARTIAL_COMPLY.value == "STYLE.PARTIAL_COMPLY"
        
        # Test all values
        all_values = [e.value for e in ComplianceStyle]
        assert len(all_values) == 11
    
    def test_harm_category_enum(self):
        """Test HarmCategory enum values."""
        assert HarmCategory.WEAPONS.value == "weapons"
        assert HarmCategory.ILLICIT_BEHAVIOR.value == "illicit_behavior"
        assert HarmCategory.CYBERCRIME.value == "cybercrime"
        assert HarmCategory.FRAUD.value == "fraud"
        assert HarmCategory.TERRORISM.value == "terrorism"
        assert HarmCategory.SELF_HARM.value == "self_harm"
        assert HarmCategory.HATE_HARASSMENT.value == "hate_harassment"
        assert HarmCategory.SEXUAL_EXPLOITATIVE.value == "sexual_exploitative"
        assert HarmCategory.ADULT_SEXUAL.value == "adult_sexual"
        assert HarmCategory.PRIVACY_PERSONAL_DATA.value == "privacy_personal_data"
        assert HarmCategory.MEDICAL_ADVICE.value == "medical_advice"
        assert HarmCategory.LEGAL_ADVICE.value == "legal_advice"
        assert HarmCategory.FINANCIAL_ADVICE.value == "financial_advice"
        assert HarmCategory.EXTREMISM.value == "extremism"
        assert HarmCategory.VIOLENCE.value == "violence"
        assert HarmCategory.BULLYING.value == "bullying"
        assert HarmCategory.CHILD_SAFETY.value == "child_safety"
        assert HarmCategory.MISINFORMATION.value == "misinformation"
        assert HarmCategory.CONTROLLED_SUBSTANCES.value == "controlled_substances"
        assert HarmCategory.GAMBLING.value == "gambling"
        assert HarmCategory.INTELLECTUAL_PROPERTY.value == "intellectual_property"
        assert HarmCategory.POLITICAL_PERSUASION.value == "political_persuasion"
        assert HarmCategory.BIOLOGICAL_THREATS.value == "biological_threats"
        assert HarmCategory.CHEMICAL_THREATS.value == "chemical_threats"
        assert HarmCategory.RADIOLOGICAL_NUCLEAR.value == "radiological_nuclear"
        assert HarmCategory.DANGEROUS_ACTIVITIES.value == "dangerous_activities"
        assert HarmCategory.OTHER.value == "other"
        
        # Test all values
        all_values = [e.value for e in HarmCategory]
        assert len(all_values) == 27
    
    def test_harmless_category_enum(self):
        """Test HarmlessCategory enum values."""
        assert HarmlessCategory.HEALTH_MEDICINE.value == "health_medicine"
        assert HarmlessCategory.EDUCATION.value == "education"
        assert HarmlessCategory.TECHNOLOGY.value == "technology"
        assert HarmlessCategory.SCIENCE_SPACE.value == "science_space"
        assert HarmlessCategory.ENVIRONMENT_CLIMATE.value == "environment_climate"
        assert HarmlessCategory.POLITICS_GOVERNMENT.value == "politics_government"
        assert HarmlessCategory.LAW_LEGAL.value == "law_legal"
        assert HarmlessCategory.BUSINESS_ENTREPRENEURSHIP.value == "business_entrepreneurship"
        assert HarmlessCategory.FINANCE_ECONOMY.value == "finance_economy"
        assert HarmlessCategory.SPORTS.value == "sports"
        assert HarmlessCategory.TRAVEL_TOURISM.value == "travel_tourism"
        assert HarmlessCategory.FOOD_COOKING.value == "food_cooking"
        assert HarmlessCategory.ENTERTAINMENT.value == "entertainment"
        assert HarmlessCategory.ARTS_LITERATURE.value == "arts_literature"
        assert HarmlessCategory.HISTORY.value == "history"
        assert HarmlessCategory.PSYCHOLOGY_SELF_DEVELOPMENT.value == "psychology_self_development"
        assert HarmlessCategory.RELIGION_PHILOSOPHY.value == "religion_philosophy"
        assert HarmlessCategory.LIFESTYLE_HOME.value == "lifestyle_home"
        assert HarmlessCategory.RELATIONSHIPS_DATING.value == "relationships_dating"
        assert HarmlessCategory.OTHER.value == "other"
        
        # Test all values
        all_values = [e.value for e in HarmlessCategory]
        assert len(all_values) == 20


class TestHeadConfig:
    """Test HeadConfig dataclass."""
    
    def test_head_config_creation(self):
        """Test HeadConfig creation."""
        config = HeadConfig(
            name="test_head",
            head_type="classification",
            class_names=["A", "B", "C"],
            num_classes=3
        )
        
        assert config.name == "test_head"
        assert config.head_type == "classification"
        assert config.class_names == ["A", "B", "C"]
        assert config.num_classes == 3
    
    def test_head_config_defaults(self):
        """Test HeadConfig with default values."""
        config = HeadConfig(
            name="test_head",
            head_type="multilabel",
            class_names=["X", "Y"]
        )
        
        assert config.name == "test_head"
        assert config.head_type == "multilabel"
        assert config.class_names == ["X", "Y"]
        assert config.num_classes is None  # Default value


class TestHeadConfigurations:
    """Test head configuration functions."""
    
    def test_get_head_configs(self):
        """Test get_head_configs function."""
        configs = get_head_configs()
        
        # Check that all expected heads are present
        expected_heads = ["head_a", "head_b_a", "head_b_b", "head_c_a", "head_c_b", "head_d"]
        for head in expected_heads:
            assert head in configs
        
        # Check head_a configuration
        head_a = configs["head_a"]
        assert head_a.head_type == "classification"
        assert head_a.num_classes == 12  # Updated count with new compliance types
        assert "REFUSAL.DIRECT" in head_a.class_names
        assert "COMPLY.BENIGN" in head_a.class_names
        assert "COMPLY.CONDITIONAL" in head_a.class_names
        assert "COMPLY.EDUCATIONAL" in head_a.class_names
        assert "COMPLY.REDIRECTED" in head_a.class_names
        assert "COMPLY.PARTIAL_SAFE" in head_a.class_names
        
        # Check head_b_a configuration
        head_b_a = configs["head_b_a"]
        assert head_b_a.head_type == "classification"
        assert head_b_a.num_classes == 7
        assert "STYLE.DIRECT" in head_b_a.class_names
        
        # Check head_b_b configuration
        head_b_b = configs["head_b_b"]
        assert head_b_b.head_type == "classification"
        assert head_b_b.num_classes == 11
        assert "STYLE.DIRECT_ANSWER" in head_b_b.class_names
        
        # Check head_c_a configuration (harm categories)
        head_c_a = configs["head_c_a"]
        assert head_c_a.head_type == "multilabel"
        assert head_c_a.num_classes == 27
        assert "weapons" in head_c_a.class_names
        assert "other" in head_c_a.class_names
        
        # Check head_c_b configuration (harmless categories)
        head_c_b = configs["head_c_b"]
        assert head_c_b.head_type == "multilabel"
        assert head_c_b.num_classes == 20
        assert "health_medicine" in head_c_b.class_names
        assert "technology" in head_c_b.class_names
        assert "other" in head_c_b.class_names
        
        # Check head_d configuration
        head_d = configs["head_d"]
        assert head_d.head_type == "boolean"
        assert head_d.num_classes == 3
        assert "prompt_harmful" in head_d.class_names
        assert "response_harmful" in head_d.class_names
        assert "response_refusal" in head_d.class_names
    
    def test_get_head_config(self):
        """Test get_head_config function."""
        # Test valid head names
        head_a = get_head_config("head_a")
        assert head_a.head_type == "classification"
        assert head_a.num_classes == 12  # Updated count
        
        head_c_a = get_head_config("head_c_a")
        assert head_c_a.head_type == "multilabel"
        assert head_c_a.num_classes == 27
        
        head_c_b = get_head_config("head_c_b")
        assert head_c_b.head_type == "multilabel"
        assert head_c_b.num_classes == 20
        
        head_d = get_head_config("head_d")
        assert head_d.head_type == "boolean"
        assert head_d.num_classes == 3
        
        # Test invalid head name - should return None or handle gracefully
        result = get_head_config("invalid_head")
        # The function should handle invalid head names gracefully
        assert result is None or isinstance(result, HeadConfig)
    
    def test_get_all_head_configs(self):
        """Test get_all_head_configs function."""
        configs = get_all_head_configs()
        
        # Should be the same as get_head_configs
        assert configs == get_head_configs()
        assert len(configs) == 6  # Updated count with head_c_a and head_c_b


class TestLabelMappings:
    """Test label mapping functions."""
    
    def test_get_label_to_id_mapping(self):
        """Test get_label_to_id_mapping function."""
        # Test head_a mapping
        mapping = get_label_to_id_mapping("head_a")
        assert "REFUSAL.DIRECT" in mapping
        assert mapping["REFUSAL.DIRECT"] == 0
        assert "COMPLY.BENIGN" in mapping
        assert mapping["COMPLY.BENIGN"] == 5
        
        # Test head_c_a mapping
        mapping = get_label_to_id_mapping("head_c_a")
        assert "weapons" in mapping
        assert mapping["weapons"] == 0
        assert "other" in mapping
        assert mapping["other"] == 26
        
        # Test head_c_b mapping
        mapping = get_label_to_id_mapping("head_c_b")
        assert "health_medicine" in mapping
        assert mapping["health_medicine"] == 0
        assert "other" in mapping
        assert mapping["other"] == 19
        
        # Test head_d mapping
        mapping = get_label_to_id_mapping("head_d")
        assert "prompt_harmful" in mapping
        assert mapping["prompt_harmful"] == 0
        assert "response_harmful" in mapping
        assert mapping["response_harmful"] == 1
        assert "response_refusal" in mapping
        assert mapping["response_refusal"] == 2
        
        # Test invalid head name - should return empty dict or handle gracefully
        result = get_label_to_id_mapping("invalid_head")
        # The function should handle invalid head names gracefully
        assert isinstance(result, dict)
    
    def test_get_id_to_label_mapping(self):
        """Test get_id_to_label_mapping function."""
        # Test head_a mapping
        mapping = get_id_to_label_mapping("head_a")
        assert mapping[0] == "REFUSAL.DIRECT"
        assert mapping[5] == "COMPLY.BENIGN"
        
        # Test head_c_a mapping
        mapping = get_id_to_label_mapping("head_c_a")
        assert mapping[0] == "weapons"
        assert mapping[26] == "other"
        
        # Test head_c_b mapping
        mapping = get_id_to_label_mapping("head_c_b")
        assert mapping[0] == "health_medicine"
        assert mapping[19] == "other"
        
        # Test head_d mapping
        mapping = get_id_to_label_mapping("head_d")
        assert mapping[0] == "prompt_harmful"
        assert mapping[1] == "response_harmful"
        assert mapping[2] == "response_refusal"
        
        # Test invalid head name - should return empty dict or handle gracefully
        result = get_id_to_label_mapping("invalid_head")
        # The function should handle invalid head names gracefully
        assert isinstance(result, dict)


class TestLabelClassification:
    """Test label classification functions."""
    
    def test_is_refusal_label(self):
        """Test is_refusal_label function."""
        # Test refusal labels
        assert is_refusal_label("REFUSAL.DIRECT") is True
        assert is_refusal_label("REFUSAL.SOFT") is True
        assert is_refusal_label("REFUSAL.PARTIAL") is True
        assert is_refusal_label("REFUSAL.CAPABILITY") is True
        assert is_refusal_label("REFUSAL.OVER") is True
        
        # Test compliance labels
        assert is_refusal_label("COMPLY.BENIGN") is False
        assert is_refusal_label("COMPLY.TRANSFORM") is False
        assert is_refusal_label("COMPLY.UNSAFE") is False
        
        # Test invalid labels
        assert is_refusal_label("INVALID.LABEL") is False
        assert is_refusal_label("") is False
    
    def test_is_compliance_label(self):
        """Test is_compliance_label function."""
        # Test compliance labels
        assert is_compliance_label("COMPLY.BENIGN") is True
        assert is_compliance_label("COMPLY.TRANSFORM") is True
        assert is_compliance_label("COMPLY.UNSAFE") is True
        assert is_compliance_label("COMPLY.CONDITIONAL") is True
        assert is_compliance_label("COMPLY.EDUCATIONAL") is True
        assert is_compliance_label("COMPLY.REDIRECTED") is True
        assert is_compliance_label("COMPLY.PARTIAL_SAFE") is True
        
        # Test refusal labels
        assert is_compliance_label("REFUSAL.DIRECT") is False
        assert is_compliance_label("REFUSAL.SOFT") is False
        assert is_compliance_label("REFUSAL.PARTIAL") is False
        assert is_compliance_label("REFUSAL.CAPABILITY") is False
        assert is_compliance_label("REFUSAL.OVER") is False
        
        # Test invalid labels
        assert is_compliance_label("INVALID.LABEL") is False
        assert is_compliance_label("") is False


class TestPrecedenceRules:
    """Test outcome precedence rules."""
    
    def test_outcome_precedence_structure(self):
        """Test that precedence rules have correct structure."""
        precedence = get_outcome_precedence()
        assert isinstance(precedence, list)
        assert len(precedence) > 0
        
        # Check that all items are OutcomeType enums
        for outcome in precedence:
            assert isinstance(outcome, OutcomeType)
    
    def test_outcome_precedence_content(self):
        """Test that precedence rules contain expected outcomes."""
        precedence = get_outcome_precedence()
        
        # Check that all outcome types are covered
        expected_outcomes = [e for e in OutcomeType]
        for outcome in expected_outcomes:
            assert outcome in precedence


class TestTaxonomyConsistency:
    """Test consistency across taxonomy components."""
    
    def test_head_configs_consistency(self):
        """Test that head configurations are consistent."""
        configs = get_head_configs()
        
        for head_name, config in configs.items():
            # Check that num_classes matches class_names length
            assert config.num_classes == len(config.class_names)
            
            # Check that class_names are not empty
            assert len(config.class_names) > 0
            
            # Check that head_type is valid
            assert config.head_type in ["classification", "multilabel", "boolean"]
    
    def test_label_mappings_consistency(self):
        """Test that label mappings are consistent."""
        configs = get_head_configs()
        
        for head_name, config in configs.items():
            # Test label to ID mapping
            label_to_id = get_label_to_id_mapping(head_name)
            id_to_label = get_id_to_label_mapping(head_name)
            
            # Check that mappings are consistent
            assert len(label_to_id) == config.num_classes
            assert len(id_to_label) == config.num_classes
            
            # Check that mappings are inverse of each other
            for label, id_val in label_to_id.items():
                assert id_to_label[id_val] == label
            
            for id_val, label in id_to_label.items():
                assert label_to_id[label] == id_val
    
    def test_enum_values_match_class_names(self):
        """Test that enum values match class names in configurations."""
        configs = get_head_configs()
        
        # Check head_a (OutcomeType)
        head_a_config = configs["head_a"]
        outcome_values = [e.value for e in OutcomeType]
        for class_name in head_a_config.class_names:
            assert class_name in outcome_values
        
        # Check head_b_a (RefusalStyle)
        head_b_a_config = configs["head_b_a"]
        refusal_values = [e.value for e in RefusalStyle]
        for class_name in head_b_a_config.class_names:
            assert class_name in refusal_values
        
        # Check head_b_b (ComplianceStyle)
        head_b_b_config = configs["head_b_b"]
        compliance_values = [e.value for e in ComplianceStyle]
        for class_name in head_b_b_config.class_names:
            assert class_name in compliance_values
        
        # Check head_c_a (HarmCategory)
        head_c_a_config = configs["head_c_a"]
        harm_values = [e.value for e in HarmCategory]
        for class_name in head_c_a_config.class_names:
            assert class_name in harm_values
        
        # Check head_c_b (HarmlessCategory)
        head_c_b_config = configs["head_c_b"]
        harmless_values = [e.value for e in HarmlessCategory]
        for class_name in head_c_b_config.class_names:
            assert class_name in harmless_values


class TestTaxonomyEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_head_name_handling(self):
        """Test that invalid head names are handled gracefully."""
        # These functions should handle invalid head names gracefully
        result1 = get_head_config("nonexistent_head")
        assert result1 is None or isinstance(result1, HeadConfig)
        
        result2 = get_label_to_id_mapping("nonexistent_head")
        assert isinstance(result2, dict)
        
        result3 = get_id_to_label_mapping("nonexistent_head")
        assert isinstance(result3, dict)
    
    def test_empty_string_labels(self):
        """Test handling of empty string labels."""
        assert is_refusal_label("") is False
        assert is_compliance_label("") is False
    
    def test_none_labels(self):
        """Test handling of None labels."""
        # These functions should handle None gracefully or raise appropriate errors
        with pytest.raises(AttributeError):
            is_refusal_label(None)
        
        with pytest.raises(AttributeError):
            is_compliance_label(None)


class TestNewComplianceTypes:
    """Test new compliance types functionality."""
    
    def test_new_compliance_types_in_precedence(self):
        """Test that new compliance types are included in precedence rules."""
        precedence = get_outcome_precedence()
        
        # Check that new compliance types are in precedence
        assert OutcomeType.COMPLY_CONDITIONAL in precedence
        assert OutcomeType.COMPLY_EDUCATIONAL in precedence
        assert OutcomeType.COMPLY_REDIRECTED in precedence
        assert OutcomeType.COMPLY_PARTIAL_SAFE in precedence
    
    def test_new_compliance_types_definitions(self):
        """Test that new compliance types have definitions."""
        definitions = get_outcome_definitions()
        
        # Check that new compliance types have definitions (using enum objects as keys)
        assert OutcomeType.COMPLY_CONDITIONAL in definitions
        assert OutcomeType.COMPLY_EDUCATIONAL in definitions
        assert OutcomeType.COMPLY_REDIRECTED in definitions
        assert OutcomeType.COMPLY_PARTIAL_SAFE in definitions
        
        # Check that definitions are not empty
        assert len(definitions[OutcomeType.COMPLY_CONDITIONAL]) > 0
        assert len(definitions[OutcomeType.COMPLY_EDUCATIONAL]) > 0
        assert len(definitions[OutcomeType.COMPLY_REDIRECTED]) > 0
        assert len(definitions[OutcomeType.COMPLY_PARTIAL_SAFE]) > 0
    
    def test_new_compliance_types_examples(self):
        """Test that new compliance types have minimal examples."""
        examples = get_minimal_examples()
        
        # Check that new compliance types have examples (using string keys)
        assert "COMPLY.CONDITIONAL" in examples
        assert "COMPLY.EDUCATIONAL" in examples
        assert "COMPLY.REDIRECTED" in examples
        assert "COMPLY.PARTIAL_SAFE" in examples
        
        # Check that examples are not empty
        assert len(examples["COMPLY.CONDITIONAL"]) > 0
        assert len(examples["COMPLY.EDUCATIONAL"]) > 0
        assert len(examples["COMPLY.REDIRECTED"]) > 0
        assert len(examples["COMPLY.PARTIAL_SAFE"]) > 0


class TestHarmlessCategories:
    """Test harmless categories functionality."""
    
    def test_harmless_category_definitions(self):
        """Test that harmless categories have definitions."""
        definitions = get_harmless_category_definitions()
        
        # Check that all harmless categories have definitions (using enum objects as keys)
        for category in HarmlessCategory:
            assert category in definitions
            assert len(definitions[category]) > 0
    
    def test_harmless_category_comprehensive_coverage(self):
        """Test that harmless categories cover major topic domains."""
        categories = [cat.value for cat in HarmlessCategory]
        
        # Check for major topic domains
        major_domains = [
            "health_medicine", "education", "technology", "science_space",
            "politics_government", "business_entrepreneurship", "finance_economy",
            "entertainment", "sports", "travel_tourism", "food_cooking"
        ]
        
        for domain in major_domains:
            assert domain in categories, f"Major domain {domain} not covered"
    
    def test_harmless_category_no_overlap_with_harm(self):
        """Test that harmless categories don't overlap with harm categories (except 'other')."""
        harm_categories = [cat.value for cat in HarmCategory]
        harmless_categories = [cat.value for cat in HarmlessCategory]
        
        # Check for no overlap (except 'other' which is expected to be in both)
        overlap = set(harm_categories) & set(harmless_categories)
        assert overlap == {"other"}, f"Unexpected overlap found between harm and harmless categories: {overlap}"


class TestUpdatedTaxonomyStructure:
    """Test the updated taxonomy structure with head_c_a and head_c_b."""
    
    def test_head_c_a_contains_harm_categories(self):
        """Test that head_c_a contains all harm categories."""
        config = get_head_config("head_c_a")
        harm_categories = [cat.value for cat in HarmCategory]
        
        for category in harm_categories:
            assert category in config.class_names
    
    def test_head_c_b_contains_harmless_categories(self):
        """Test that head_c_b contains all harmless categories."""
        config = get_head_config("head_c_b")
        harmless_categories = [cat.value for cat in HarmlessCategory]
        
        for category in harmless_categories:
            assert category in config.class_names
    
    def test_head_c_a_and_c_b_are_separate(self):
        """Test that head_c_a and head_c_b are separate (except for 'other' category)."""
        config_a = get_head_config("head_c_a")
        config_b = get_head_config("head_c_b")
        
        # Only 'other' should overlap between the two heads
        overlap = set(config_a.class_names) & set(config_b.class_names)
        assert overlap == {"other"}, f"Unexpected overlap found between head_c_a and head_c_b: {overlap}"
    
    def test_head_c_a_and_c_b_both_multilabel(self):
        """Test that both head_c_a and head_c_b are multilabel."""
        config_a = get_head_config("head_c_a")
        config_b = get_head_config("head_c_b")
        
        assert config_a.head_type == "multilabel"
        assert config_b.head_type == "multilabel"
    
    def test_head_c_a_and_c_b_correct_sizes(self):
        """Test that head_c_a and head_c_b have correct sizes."""
        config_a = get_head_config("head_c_a")
        config_b = get_head_config("head_c_b")
        
        assert config_a.num_classes == len(HarmCategory)
        assert config_b.num_classes == len(HarmlessCategory)
        assert config_a.num_classes == 27
        assert config_b.num_classes == 20
